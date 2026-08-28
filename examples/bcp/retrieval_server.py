"""
BrowseComp-Plus retrieval server — Qwen3-Embedding dense retrieval.

Usage:
    python examples/bcp/retrieval_server.py --port 9000

    # First-time setup (no cache): encodes all documents on GPU, saves cache.
    # Subsequent runs load the cache and encode every online query on GPU.

Endpoints:
    POST /retrieve  {"queries": [...], "topk": 3}
    POST /embed     {"texts": [...], "instruction": ""}   # for reward semantic judge
    POST /get_doc   {"docid": "..."}
    POST /get_doc_chunks {"docid": "...", "query": "...", "topk": 3}
    GET  /health
"""

import argparse
import asyncio
import os
import pickle
import re
import sys
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def extract_title(text: str, url: str = "") -> str:
    """Extract the title from a document's YAML front-matter or derive it from the URL."""
    m = re.search(r"^---\s*\ntitle:\s*(.+?)\n", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    if url:
        return url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return "Unknown"


def build_corpus(data_dir: str):
    """Load all unique documents from parquet files in *data_dir* and return a docid→doc mapping."""
    import pandas as pd

    print("Loading parquet files ...", flush=True)
    dfs = []
    for split in ["train.parquet", "test.parquet"]:
        path = f"{data_dir}/{split}"
        try:
            dfs.append(pd.read_parquet(path))
        except FileNotFoundError:
            print(f"  Warning: {path} not found, skipping.", flush=True)
    if not dfs:
        raise RuntimeError(f"No parquet files found in {data_dir}")
    df = pd.concat(dfs, ignore_index=True)

    seen = {}
    for col in ["gold_docs", "negative_docs", "evidence_docs"]:
        if col not in df.columns:
            continue
        for row_docs in df[col].dropna():
            if row_docs is None:
                continue
            for doc in row_docs:
                if not isinstance(doc, dict):
                    continue
                docid = str(doc.get("docid", ""))
                if docid and docid not in seen:
                    seen[docid] = doc

    print(f"  Loaded {len(seen)} unique documents.", flush=True)
    return seen


# ---------------------------------------------------------------------------
# Dense retriever (Qwen3-Embedding + FAISS)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever backed by Qwen3-Embedding and a FAISS flat inner-product index."""

    def __init__(
        self,
        data_dir: str,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        gpu_id: int = 7,
        cache_path: str | None = None,
        batch_size: int = 4,
        max_doc_length: int = 8192,
    ):
        """Initialize the retriever.

        Loads the embedding model and either reads a pre-built FAISS index from
        *cache_path* or encodes all documents from scratch on GPU and writes the
        cache. Online query encoding always runs on the selected GPU.
        """
        self._device_str = f"cuda:{gpu_id}"  # lazy: don't create a CUDA context yet
        self._model_name = model_name
        self.batch_size = batch_size
        self.max_doc_length = max_doc_length
        self._model_on_gpu = False
        self.device = None                 # set lazily on first GPU use
        self.model = None                  # set lazily or eagerly depending on cache

        has_cache = cache_path and os.path.exists(cache_path)

        if has_cache:
            # Cache exists: stage the model without creating a CUDA context.
            from transformers import AutoTokenizer, AutoModel
            import torch

            print(f"Loading embedding model {model_name} on cpu (cache hit) ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, dtype=torch.float32, trust_remote_code=True
            ).to("cpu").eval()
            print("  Model loaded on CPU.", flush=True)

            print(f"Loading dense index from cache: {cache_path} ...", flush=True)
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.docids = cached["docids"]
            self.contents = cached["contents"]
            embeddings = cached["embeddings"]  # (N, D) float32
            self._build_faiss(embeddings)
            print(f"  Loaded {len(self.docids)} documents from cache.", flush=True)
        else:
            # No cache: need GPU for document encoding — use ALL GPUs for speed
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.device = torch.device(self._device_str)

            print(f"Loading embedding model {model_name} on ALL GPUs for encoding ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, dtype=torch.float16, trust_remote_code=True
            ).eval()

            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                # Wrapper: only return last_hidden_state to avoid gathering KV cache (OOM)
                class _HiddenStateOnly(torch.nn.Module):
                    """Wrap an embedding model and expose only `last_hidden_state`."""

                    def __init__(self, base):
                        """Store the underlying Hugging Face model."""
                        super().__init__()
                        self.base = base

                    def forward(self, **kwargs):
                        """Run the base model and return only the hidden states tensor."""
                        return self.base(**kwargs).last_hidden_state
                self._raw_model = self.model
                self.model = torch.nn.DataParallel(_HiddenStateOnly(self.model)).cuda()
                self._dp_device = torch.device("cuda:0")
                print(f"  Using DataParallel on {n_gpus} GPUs.", flush=True)
            else:
                self.model = self.model.to(self.device)
                self._dp_device = self.device
                print(f"  Single GPU: {self._device_str}", flush=True)
            self._model_on_gpu = True
            # Use larger batch size with multi-GPU
            self._init_batch_size = batch_size * n_gpus

            doc_map = build_corpus(data_dir)
            self.docids = list(doc_map.keys())
            docs = [doc_map[d] for d in self.docids]

            self.contents = []
            raw_texts = []
            for doc in docs:
                title = extract_title(doc.get("text", ""), doc.get("url", ""))
                body = doc.get("text", "")
                self.contents.append(f"{title}\n{body}")
                raw_texts.append(body)

            print(f"Encoding {len(raw_texts)} documents (batch_size={self._init_batch_size}) ...", flush=True)
            embeddings = self._encode_bulk(raw_texts, max_length=max_doc_length)
            self._build_faiss(embeddings)

            if cache_path:
                print(f"Saving dense cache to {cache_path} ...", flush=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "docids": self.docids,
                        "contents": self.contents,
                        "embeddings": embeddings,
                    }, f)

            # Offload: restore raw model, move to CPU for online queries
            if hasattr(self, '_raw_model'):
                self.model = self._raw_model
                del self._raw_model
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._model_on_gpu = False

        print("Dense retriever ready.", flush=True)

    # ------------------------------------------------------------------
    def _ensure_model_on_gpu(self):
        """Move model to GPU if not already there. Lazily creates CUDA device on first call."""
        import torch
        if self.device is None:
            self.device = torch.device(self._device_str)
        if not self._model_on_gpu:
            self.model = self.model.half().to(self.device)
            self._model_on_gpu = True

    # ------------------------------------------------------------------
    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last non-padding token as the sequence embedding (Qwen3 decoder style)."""
        import torch
        # Qwen3-Embedding uses decoder architecture -> last token pooling
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]

    # ------------------------------------------------------------------
    def _encode_bulk(self, texts: List[str], max_length: int = 8192) -> np.ndarray:
        """Multi-GPU bulk encoding for initial document indexing."""
        import torch
        import torch.nn.functional as F

        device = self._dp_device
        bs = self._init_batch_size
        all_embs = []
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            enc = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                hidden = self.model(**enc)  # wrapper returns last_hidden_state directly
            emb = self._last_token_pool(hidden, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
            if (i // bs) % 20 == 0:
                print(f"  Encoded {i + len(batch)}/{len(texts)}", flush=True)
        return np.vstack(all_embs)

    def _encode(self, texts: List[str], instruction: Optional[str], max_length: int = 512) -> np.ndarray:
        """Encode *texts* on GPU (moved lazily) and return L2-normalized embeddings."""
        import torch
        import torch.nn.functional as F

        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        self._ensure_model_on_gpu()
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._last_token_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
            if (i // self.batch_size) % 50 == 0:
                print(f"  Encoded {i + len(batch)}/{len(texts)}", flush=True)
        # Keep model on GPU after first query for fast subsequent queries.
        # Model is staged on CPU at startup and stays on GPU after the first query.
        return np.vstack(all_embs)

    def _build_faiss(self, embeddings: np.ndarray):
        """Build a FAISS IndexFlatIP from L2-normalized *embeddings* and store it in self.index."""
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity (vectors are L2-normalized)
        index.add(embeddings)
        self.index = index
        self.dim = dim
        print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}", flush=True)

    # ------------------------------------------------------------------
    def search(self, query: str, topk: int = 3) -> List[dict]:
        """Search for a single query."""
        query_embs = self._encode(
            [query],
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            max_length=256,
        )  # (1, D)
        scores, indices = self.index.search(query_embs, topk)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "docid": self.docids[idx],
                "document": {"contents": self.contents[idx]},
                "score": float(score),
            })
        return results

    def embed(self, texts: List[str], instruction: str = "") -> np.ndarray:
        """Public method used by /embed endpoint (for reward semantic judge)."""
        return self._encode(texts, instruction=instruction or None, max_length=256)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()
retriever = None
_search_lock = asyncio.Lock()


class RetrieveRequest(BaseModel):
    """Request body for the POST /retrieve endpoint."""

    queries: List[str]
    topk: int = 3
    return_scores: bool = True


class EmbedRequest(BaseModel):
    """Request body for the POST /embed endpoint."""

    texts: List[str]
    instruction: str = ""


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """Run dense retrieval for each query and return ranked document lists."""
    # Query encoding uses GPU; serialize requests to avoid embedding-model OOM.
    async with _search_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: [retriever.search(q, req.topk) for q in req.queries],
        )
    return {"result": result}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Return L2-normalized embeddings for reward semantic judgment."""
    if not hasattr(retriever, "embed"):
        return {"error": "embed not supported", "embeddings": None}
    async with _search_lock:
        loop = asyncio.get_event_loop()
        embs = await loop.run_in_executor(
            None, lambda: retriever.embed(req.texts, req.instruction)
        )
    return {"embeddings": embs.tolist()}


class GetDocRequest(BaseModel):
    """Request body for the POST /get_doc endpoint."""

    docid: str


@app.post("/get_doc")
async def get_doc(req: GetDocRequest):
    """Return full document content by docid."""
    if retriever is None:
        return {"error": "retriever not initialized", "document": None}
    try:
        idx = retriever.docids.index(req.docid)
    except ValueError:
        return {"error": f"docid '{req.docid}' not found", "document": None}
    return {"document": {"contents": retriever.contents[idx]}, "docid": req.docid}


# ---------------------------------------------------------------------------
# Sub-document chunk retrieval (second-stage retrieval for open_page)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 512       # tokens per chunk (rough: 1 token ≈ 4 chars)
_CHUNK_OVERLAP = 64     # overlap tokens between adjacent chunks
_CHARS_PER_TOKEN = 4


def _tokenize(text: str) -> List[str]:
    """Lowercase and split *text* into word tokens, stripping punctuation."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _split_into_chunks(text: str, chunk_size: int = _CHUNK_SIZE,
                       overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks by approximate token count."""
    char_chunk = chunk_size * _CHARS_PER_TOKEN
    char_overlap = overlap * _CHARS_PER_TOKEN
    step = max(char_chunk - char_overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start: start + char_chunk]
        if chunk.strip():
            chunks.append(chunk)
        if start + char_chunk >= len(text):
            break
    return chunks


class GetDocChunksRequest(BaseModel):
    """Request body for the POST /get_doc_chunks endpoint."""

    docid: str
    query: str
    topk: int = 3
    chunk_size: int = _CHUNK_SIZE
    chunk_overlap: int = _CHUNK_OVERLAP


def _bm25_rank_chunks(query: str, chunks: List[str], topk: int) -> List[tuple]:
    """Rank *chunks* against *query* using BM25.

    Returns a list of (chunk_index, score, chunk_text) sorted by score desc.
    No GPU, no lock, runs in < 10 ms.
    """
    from math import log

    # Tokenize
    q_tokens = _tokenize(query)
    if not q_tokens:
        # Fallback: return first topk chunks in order
        return [(i, 0.0, chunks[i]) for i in range(min(topk, len(chunks)))]

    chunk_tokens = [_tokenize(c) for c in chunks]
    n = len(chunks)
    avg_dl = sum(len(ct) for ct in chunk_tokens) / max(n, 1)

    # BM25 parameters
    k1 = 1.2
    b = 0.75

    # Document frequency for query terms
    df = {}
    for t in q_tokens:
        df[t] = sum(1 for ct in chunk_tokens if t in set(ct))

    # Score each chunk
    scores = []
    for i, ct in enumerate(chunk_tokens):
        tf_map = {}
        for t in ct:
            tf_map[t] = tf_map.get(t, 0) + 1
        dl = len(ct)
        score = 0.0
        for t in q_tokens:
            if t not in tf_map:
                continue
            tf = tf_map[t]
            idf = log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        scores.append((i, score, chunks[i]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]


@app.post("/get_doc_chunks")
async def get_doc_chunks(req: GetDocChunksRequest):
    """Sub-document retrieval: split a document into chunks and rank them
    against *query* using BM25.  No GPU needed — does not block /retrieve.
    """
    if retriever is None:
        return {"error": "retriever not initialized"}

    # Locate the document
    try:
        idx = retriever.docids.index(req.docid)
    except ValueError:
        return {"error": f"docid '{req.docid}' not found"}

    full_text = retriever.contents[idx]
    title = full_text.split("\n")[0]
    body = "\n".join(full_text.split("\n")[1:])

    # Split into chunks
    chunks = _split_into_chunks(body, req.chunk_size, req.chunk_overlap)
    if not chunks:
        return {"error": "document body is empty", "docid": req.docid}

    # BM25 ranking — pure CPU, no lock, < 10 ms
    ranked = _bm25_rank_chunks(req.query, chunks, min(req.topk, len(chunks)))

    results = []
    for chunk_idx, score, text in ranked:
        results.append({
            "chunk_index": chunk_idx,
            "score": round(score, 4),
            "text": text,
        })

    return {
        "docid": req.docid,
        "title": title,
        "total_chunks": len(chunks),
        "chunks": results,
    }


@app.get("/health")
def health():
    """Return server health status and the number of indexed documents."""
    n = len(retriever.docids) if retriever else 0
    return {"status": "ok", "mode": "dense", "num_docs": n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments, initialise the retriever, and start the uvicorn server."""
    global retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu_id", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dense_cache", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        parser.error(f"--data_dir is not a directory: {args.data_dir}")
    if not os.path.isdir(args.model):
        parser.error(f"--model is not a directory: {args.model}")
    if not os.path.isfile(args.dense_cache):
        parser.error(f"--dense_cache is not a file: {args.dense_cache}")

    try:
        import faiss  # noqa: F401
    except ImportError:
        print("ERROR: faiss not installed. Run:", flush=True)
        print("  pip install faiss-cpu", flush=True)
        sys.exit(1)

    import torch

    if not torch.cuda.is_available():
        parser.error("BCP retrieval requires CUDA; no GPU is available")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        parser.error(
            f"--gpu_id must be in [0, {torch.cuda.device_count() - 1}], got {args.gpu_id}"
        )

    retriever = DenseRetriever(
        args.data_dir,
        model_name=args.model,
        gpu_id=args.gpu_id,
        cache_path=args.dense_cache,
        batch_size=args.batch_size,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
