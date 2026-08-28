"""Convert FaissSearcher shard pkl files into the dense_cache.pkl format used by retrieval_server.py.

FaissSearcher shard format : (embeddings: ndarray(N, D), docid_lookup: list[int])
Target dense_cache format  : {"docids": list[str], "contents": list[str], "embeddings": ndarray}

Usage:
    python3 examples/bcp/build_dense_cache.py \\
        --index-path "/data/browsecomp-plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \\
        --corpus-path /data/browsecomp-plus/corpus/data \\
        --output /data/browsecomp-plus/browsecomp_dense_cache.pkl
"""
import argparse
import glob
import os
import pickle

import numpy as np
from datasets import load_dataset


def main():
    """Merge FaissSearcher shard pkl files and convert them into a dense_cache.pkl."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True, help="Glob pattern for shard pkl files")
    parser.add_argument("--corpus-path", required=True, help="Local parquet corpus directory")
    parser.add_argument("--output", required=True, help="Output dense_cache.pkl path")
    args = parser.parse_args()

    # 1. Load shards
    shard_paths = sorted(glob.glob(args.index_path))
    if not shard_paths:
        parser.error(f"--index-path matched no files: {args.index_path}")

    corpus_paths = sorted(glob.glob(os.path.join(args.corpus_path, "*.parquet")))
    if not corpus_paths:
        parser.error(f"--corpus-path contains no Parquet files: {args.corpus_path}")

    all_embeddings, all_docids_int = [], []
    for path in shard_paths:
        with open(path, "rb") as f:
            reps, lookup = pickle.load(f)
        all_embeddings.append(np.array(reps))
        all_docids_int.extend(lookup)
        print(f"  Loaded {len(lookup)} docs from {path}")
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    if len(embeddings) != len(all_docids_int):
        raise ValueError(
            "Embedding and docid counts differ: "
            f"{len(embeddings)} embeddings vs {len(all_docids_int)} docids"
        )
    print(f"Total embeddings: {embeddings.shape}")

    # 2. Load corpus: docid -> text
    print(f"Loading corpus from {args.corpus_path} ...")
    ds = load_dataset(
        "parquet",
        data_files={"train": corpus_paths},
        split="train",
    )
    docid_to_text = {str(row["docid"]): row["text"] for row in ds}
    print(f"  Corpus: {len(docid_to_text)} documents")

    # 3. Build contents (title + "\n" + body) in shard order
    docids_str, contents = [], []
    missing = 0
    for did_int in all_docids_int:
        did = str(did_int)
        text = docid_to_text.get(did, "")
        if not text:
            missing += 1
        lines = text.split("\n")
        title = lines[0] if lines else ""
        body = "\n".join(lines[1:]) if len(lines) > 1 else text
        docids_str.append(did)
        contents.append(f"{title}\n{body}")
    if missing:
        raise ValueError(f"{missing} index docids had no text in the corpus")

    # 4. Save
    print(f"Saving dense cache to {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"docids": docids_str, "contents": contents, "embeddings": embeddings}, f)
    print(f"Done. Saved {len(docids_str)} docs, embeddings shape={embeddings.shape}")


if __name__ == "__main__":
    main()
