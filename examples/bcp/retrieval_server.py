"""
BrowseComp-Plus retrieval server — Qwen3-Embedding dense retrieval on CPU.

Usage:
    python examples/bcp/retrieval_server.py \
        --model /path/to/Qwen3-Embedding-8B \
        --dense_cache /path/to/browsecomp_dense_cache.pkl \
        --port 9000

    The dense cache must be built beforehand with examples/bcp/build_dense_cache.py.
    Both the index and online query encoding live entirely on CPU.

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
# Dense retriever (Qwen3-Embedding + FAISS)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever backed by Qwen3-Embedding on CPU and a FAISS flat inner-product index."""

    def __init__(
        self,
        cache_path: str,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 4,
    ):
        """Load the embedding model on CPU and build the FAISS index from *cache_path*.

        *cache_path* must point to a cache produced by
        ``examples/bcp/build_dense_cache.py``.
        """
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.batch_size = batch_size

        print(f"Loading embedding model {model_name} on CPU ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, dtype=torch.float32, trust_remote_code=True
        ).eval()

        print(f"Loading dense index from cache: {cache_path} ...", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        self.docids = cached["docids"]
        self.contents = cached["contents"]
        self._build_faiss(cached["embeddings"])
        print(f"  Loaded {len(self.docids)} documents from cache.", flush=True)
        print("Dense retriever ready.", flush=True)

    # ------------------------------------------------------------------
    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last non-padding token as the sequence embedding (Qwen3 decoder style)."""
        import torch
        # Qwen3-Embedding uses decoder architecture -> last token pooling
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]

    # ------------------------------------------------------------------
    def _encode(self, texts: List[str], instruction: Optional[str], max_length: int = 512) -> np.ndarray:
        """Encode *texts* on CPU and return L2-normalized embeddings."""
        import torch
        import torch.nn.functional as F

        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._last_token_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.numpy())
            if (i // self.batch_size) % 50 == 0:
                print(f"  Encoded {i + len(batch)}/{len(texts)}", flush=True)
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
    # Serialize requests: one query encoding at a time through the shared model.
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
    No model forward, no lock, runs in < 10 ms.
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
    against *query* using BM25.  No model forward — does not block /retrieve.
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dense_cache", required=True,
                        help="Cache built by examples/bcp/build_dense_cache.py")
    args = parser.parse_args()

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

    retriever = DenseRetriever(
        args.dense_cache,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
