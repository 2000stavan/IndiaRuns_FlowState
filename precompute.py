#!/usr/bin/env python3
"""
precompute.py — Run once offline. No time limit. ~3-5 hours for 100K.
Generates: FAISS index, feature cache.

Usage:
  python precompute.py --candidates data/candidates.jsonl.gz
  python precompute.py --candidates data/candidates.jsonl
  python precompute.py --candidates data/sample_candidates.json  # dev mode
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from scoring import (
    detect_honeypot,
    compute_career_arc_score,
    compute_skill_trust_score,
    compute_location_score,
    compute_yoe_score,
    compute_behavioral_score,
)

# =============================================================================
# Configuration
# =============================================================================

JD_TEXT = (
    "Senior AI Engineer founding team at Redrob AI, Pune or Noida India. "
    "Requirements: production embedding-based retrieval systems — sentence-transformers, "
    "BGE, E5, OpenAI embeddings — handling drift, index refresh, quality regression. "
    "Production vector databases: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
    "Elasticsearch, FAISS. Strong Python. Ranking evaluation frameworks: "
    "NDCG, MRR, MAP, offline-to-online correlation, A/B testing. "
    "5-9 years applied ML at product companies. NOT consulting firms. NOT pure research. "
    "Has shipped ranking, search, or recommendation systems at meaningful scale. "
    "Location: Pune, Noida preferred; also Hyderabad, Bangalore, Delhi NCR, Mumbai. "
    "Notice: sub-30 days ideal, 60 days acceptable. "
    "Nice-to-have: LLM fine-tuning LoRA QLoRA PEFT, learning-to-rank, open-source contributions."
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


# =============================================================================
# Data Loading (robust: .gz → .jsonl → .json fallback)
# =============================================================================

def load_candidates(path: str) -> list:
    """
    Load candidates from .gz, .jsonl, or .json file.
    Falls back gracefully if the exact file doesn't exist.
    """
    p = Path(path)

    # Try the exact path first
    if p.exists():
        if path.endswith(".gz"):
            print(f"  Loading gzipped JSONL: {path}")
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif path.endswith(".jsonl"):
            print(f"  Loading JSONL: {path}")
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        else:
            print(f"  Loading JSON array: {path}")
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    # Fallback: try alternate extensions
    fallbacks = []
    if path.endswith(".gz"):
        fallbacks = [path[:-3], path[:-8] + ".json"]  # .jsonl, .json
    elif path.endswith(".jsonl"):
        fallbacks = [path + ".gz", path[:-6] + ".json"]  # .gz, .json
    else:
        fallbacks = [path + ".jsonl", path + ".jsonl.gz"]

    for fb in fallbacks:
        if Path(fb).exists():
            print(f"  File {path} not found, falling back to: {fb}")
            return load_candidates(fb)

    raise FileNotFoundError(f"Cannot find candidates file: {path} (also tried: {fallbacks})")


# =============================================================================
# Candidate Text Construction
# =============================================================================

def build_candidate_text(candidate: dict) -> str:
    """
    Build text for embedding. Embeds CAREER DESCRIPTIONS — not skills[*].name.
    Uses BGE-required prefix for passage encoding.
    """
    p = candidate["profile"]

    # Career descriptions are the PRIMARY signal
    parts = []
    for job in candidate["career_history"]:
        if job.get("description"):
            parts.append(f"{job['title']} at {job['company']}: {job['description']}")

    career_text = " ".join(parts)
    full = f"{p['headline']}. {p['summary']}. {career_text}"

    # BGE prefix for passage/document encoding
    return "Represent this professional background for retrieval: " + full[:3000]


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Pre-compute embeddings and features")
    ap.add_argument("--candidates", required=True, help="Path to candidates file")
    args = ap.parse_args()

    Path("embeddings").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load candidates
    # -------------------------------------------------------------------------
    print("[1/5] Loading candidates...")
    candidates = load_candidates(args.candidates)
    cids = [c["candidate_id"] for c in candidates]
    print(f"  {len(candidates):,} candidates loaded")

    # -------------------------------------------------------------------------
    # Step 2: Compute features for all candidates
    # -------------------------------------------------------------------------
    print("[2/5] Computing features for all candidates...")
    features = {}
    hp_count = 0

    for c in tqdm(candidates, desc="  Features"):
        cid = c["candidate_id"]
        is_hp = detect_honeypot(c)
        if is_hp:
            hp_count += 1

        features[cid] = {
            "is_honeypot":       is_hp,
            "career_arc_score":  compute_career_arc_score(c),
            "skill_trust_score": compute_skill_trust_score(c),
            "location_score":    compute_location_score(c),
            "yoe_score":         compute_yoe_score(c),
            "behavioral_score":  compute_behavioral_score(c["redrob_signals"]),
        }

    print(f"  {hp_count} honeypots detected out of {len(candidates):,}")

    # Save features and candidate ID list
    with open("cache/features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open("cache/cids.pkl", "wb") as f:
        pickle.dump(cids, f)
    print("  Features saved to cache/")

    # -------------------------------------------------------------------------
    # Step 3: Load embedding model
    # -------------------------------------------------------------------------
    print("[3/5] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model: {MODEL_NAME}")

    # -------------------------------------------------------------------------
    # Step 4: Embed JD
    # -------------------------------------------------------------------------
    print("[4/5] Embedding JD...")
    jd_query = "Retrieve candidates matching this job description: " + JD_TEXT
    jd_emb = model.encode(
        jd_query,
        normalize_embeddings=True,
    ).astype("float32")
    np.save("embeddings/jd.npy", jd_emb)
    print(f"  JD embedding shape: {jd_emb.shape}")

    # -------------------------------------------------------------------------
    # Step 5: Embed all candidates and build FAISS index
    # -------------------------------------------------------------------------
    print(f"[5/5] Embedding {len(candidates):,} candidates (batch={BATCH_SIZE})...")
    texts = [build_candidate_text(c) for c in candidates]

    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        all_embs.append(emb)

    embs = np.vstack(all_embs)
    print(f"  Embeddings shape: {embs.shape}")

    # Build FAISS index (inner product = cosine similarity on normalized vectors)
    print("  Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, "embeddings/index.bin")
    print(f"  Index: {index.ntotal:,} vectors, dim={embs.shape[1]}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Pre-computation complete!")
    print(f"  Candidates:  {len(candidates):,}")
    print(f"  Honeypots:   {hp_count}")
    print(f"  Embeddings:  {embs.shape}")
    print(f"  FAISS index: embeddings/index.bin")
    print(f"  Features:    cache/features.pkl")
    print(f"  CIDs:        cache/cids.pkl")
    print(f"  JD emb:      embeddings/jd.npy")
    print("=" * 60)
    print("Ready to run: python rank.py --candidates <path> --out output/team_xxx.csv")


if __name__ == "__main__":
    main()
