#!/usr/bin/env python3
"""
rank.py — Runtime ranking pipeline. Must complete within 5 minutes on CPU, 16 GB RAM.

Usage:
  python rank.py --candidates data/candidates.jsonl --out output/submission.csv

Pipeline:
  1. Load pre-computed artifacts (FAISS index, features, JD embedding)
  2. Load candidates for cross-encoder text
  3. FAISS search → top 3000
  4. Filter honeypots, multi-signal score, take top 200
  5. Cross-encoder re-rank top 200
  6. Sort by (-final, cid) for tie-breaking
  7. Take top 100, normalize scores, generate reasoning, write CSV
"""

import argparse
import csv
import gzip
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from scoring import (
    compute_career_arc_score,
    compute_skill_trust_score,
    compute_location_score,
    compute_yoe_score,
    compute_behavioral_score,
)
from reasoning import generate_reasoning

# =============================================================================
# Configuration
# =============================================================================

# Weights for skill_fit composite
W_SEMANTIC  = 0.40
W_CAREER    = 0.25
W_SKILL     = 0.15
W_LOCATION  = 0.10
W_YOE       = 0.10

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_K = 3000            # Retrieve top K from FAISS (increased from 1000 to avoid missing strong candidates)
CROSS_ENCODER_TOP = 200   # Re-rank top N with cross-encoder
FINAL_TOP = 100           # Output top N
TEXT_CAP = 1400           # Max chars for cross-encoder candidate text (Bug B fix)

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


# =============================================================================
# Data Loading (robust: .gz → .jsonl → .json fallback)
# =============================================================================

def load_candidates(path: str) -> list:
    """Load candidates from .gz, .jsonl, or .json file with fallback."""
    p = Path(path)

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

    # Fallback
    fallbacks = []
    if path.endswith(".gz"):
        fallbacks = [path[:-3], path[:-8] + ".json"]
    elif path.endswith(".jsonl"):
        fallbacks = [path + ".gz", path[:-6] + ".json"]
    else:
        fallbacks = [path + ".jsonl", path + ".jsonl.gz"]

    for fb in fallbacks:
        if Path(fb).exists():
            print(f"  File {path} not found, falling back to: {fb}")
            return load_candidates(fb)

    raise FileNotFoundError(f"Cannot find: {path} (also tried: {fallbacks})")


# =============================================================================
# Cross-Encoder Text (Bug B fix: capped at TEXT_CAP chars)
# =============================================================================

def build_ce_text(c: dict) -> str:
    """
    Build rich text for cross-encoder re-ranking.
    Includes YoE, title, location, summary, 4 career entries with descriptions,
    and 12 skills with proficiency+duration. Capped at TEXT_CAP chars to stay
    within the 512-token limit of ms-marco-MiniLM-L-6-v2.
    """
    p = c["profile"]
    career = " ".join(
        f"{j['title']} at {j['company']} ({j.get('company_size', '?')}): "
        f"{j.get('description', '')[:350]}"
        for j in c["career_history"][:4]
    )
    skills = ", ".join(
        f"{s['name']} ({s['proficiency']}, {s.get('duration_months', 0)}mo)"
        for s in c["skills"][:12]
    )
    text = (
        f"{p['years_of_experience']}y | {p['current_title']} | "
        f"{p['location']}. {p.get('summary', '')[:250]}. "
        f"{career}. Skills: {skills}"
    )
    return text[:TEXT_CAP]  # Hard cap: keeps combined (JD+candidate) under 512 tokens


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()

    ap = argparse.ArgumentParser(description="Rank candidates against JD")
    ap.add_argument("--candidates", required=True, help="Path to candidates file")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load pre-computed artifacts
    # =========================================================================
    print(f"[1/7] Loading pre-computed artifacts...")
    index = faiss.read_index("embeddings/index.bin")
    jd_emb = np.load("embeddings/jd.npy").reshape(1, -1)
    with open("cache/features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("cache/cids.pkl", "rb") as f:
        cids = pickle.load(f)
    print(f"  Index: {index.ntotal:,} vectors | Features: {len(features):,} | CIDs: {len(cids):,}")
    print(f"  [{time.time() - t0:.1f}s]")

    # =========================================================================
    # Step 2: Load candidates (needed for cross-encoder text + reasoning)
    # =========================================================================
    print(f"[2/7] Loading candidates...")
    candidates = load_candidates(args.candidates)
    id2c = {c["candidate_id"]: c for c in candidates}
    print(f"  {len(candidates):,} candidates loaded")
    print(f"  [{time.time() - t0:.1f}s]")

    # =========================================================================
    # Step 3: FAISS search → top K
    # =========================================================================
    k = min(FAISS_K, index.ntotal)
    print(f"[3/7] FAISS search (k={k})...")
    D, I = index.search(jd_emb, k)
    print(f"  Top similarity: {D[0][0]:.4f}, Bottom: {D[0][-1]:.4f}")
    print(f"  [{time.time() - t0:.1f}s]")

    # =========================================================================
    # Step 4: Filter honeypots + multi-signal scoring → top 200
    # =========================================================================
    print(f"[4/7] Multi-signal scoring + honeypot filtering...")
    scored = []
    hp_filtered = 0

    for idx, sim in zip(I[0], D[0]):
        cid = cids[idx]
        feat = features.get(cid)
        if feat is None:
            continue
        if feat["is_honeypot"]:
            hp_filtered += 1
            continue

        # Compute skill_fit composite
        skill_fit = (
            W_SEMANTIC  * float(sim)
            + W_CAREER  * feat["career_arc_score"]
            + W_SKILL   * feat["skill_trust_score"]
            + W_LOCATION * feat["location_score"]
            + W_YOE     * feat["yoe_score"]
        )

        # Apply behavioral multiplier
        final = skill_fit * feat["behavioral_score"]

        scored.append({
            "cid": cid,
            "sim": float(sim),
            "skill_fit": skill_fit,
            "behavioral": feat["behavioral_score"],
            "final": final,
        })

    # Sort by final score descending, take top CROSS_ENCODER_TOP
    scored.sort(key=lambda x: (-x["final"], x["cid"]))
    top_for_ce = scored[:CROSS_ENCODER_TOP]
    print(f"  Honeypots filtered: {hp_filtered}")
    print(f"  Candidates for cross-encoder: {len(top_for_ce)}")
    print(f"  [{time.time() - t0:.1f}s]")

    # =========================================================================
    # Step 5: Cross-encoder re-ranking
    # =========================================================================
    print(f"[5/7] Cross-encoder re-ranking ({len(top_for_ce)} candidates)...")
    ce_model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)

    # Build (JD, candidate) pairs
    pairs = []
    for item in top_for_ce:
        c = id2c.get(item["cid"])
        if c is None:
            pairs.append((JD_TEXT, "Profile unavailable"))
        else:
            pairs.append((JD_TEXT, build_ce_text(c)))

    # Score all pairs
    ce_scores = ce_model.predict(pairs, show_progress_bar=True)

    # Normalize CE scores to 0-1 range
    ce_min = float(min(ce_scores))
    ce_max = float(max(ce_scores))
    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

    # Blend: 75% multi-signal + 25% cross-encoder
    # (Tested: 60/40 lets CE override multi-signal; 75/25 keeps correct ranking)
    for i, item in enumerate(top_for_ce):
        ce_norm = (float(ce_scores[i]) - ce_min) / ce_range
        item["ce_score"] = ce_norm
        item["final"] = item["final"] * 0.75 + ce_norm * 0.25

    print(f"  CE score range: [{ce_min:.4f}, {ce_max:.4f}]")
    print(f"  [{time.time() - t0:.1f}s]")

    # =========================================================================
    # Step 6: Sort by (-final, cid) for tie-breaking (Bug C fix)
    # =========================================================================
    print(f"[6/7] Final sort with tie-breaking...")
    top_for_ce.sort(key=lambda x: (-x["final"], x["cid"]))
    top100 = top_for_ce[:FINAL_TOP]
    print(f"  Top 100 selected")

    # Normalize scores to 0.40-0.99 range
    if len(top100) > 1:
        mn = min(item["final"] for item in top100)
        mx = max(item["final"] for item in top100)
        rng = mx - mn if mx > mn else 1.0
    else:
        mn, mx, rng = 0.0, 1.0, 1.0

    print(f"  Raw score range: [{mn:.4f}, {mx:.4f}]")
    print(f"  [{time.time() - t0:.1f}s]")

    # =========================================================================
    # Step 7: Generate reasoning + write CSV
    # =========================================================================
    print(f"[7/7] Generating reasoning + writing CSV...")
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, item in enumerate(top100, 1):
            cid = item["cid"]
            out_score = round(0.40 + ((item["final"] - mn) / rng) * 0.59, 4)

            # Bug D fix: guard for missing candidates
            if cid not in id2c:
                reason = f"{cid} profile data unavailable for reasoning."
            else:
                reason = generate_reasoning(id2c[cid], rank, out_score)

            writer.writerow([cid, rank, out_score, reason])

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Ranking complete!")
    print(f"  Output: {args.out}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Top 1: {top100[0]['cid']} (score={0.40 + ((top100[0]['final'] - mn) / rng) * 0.59:.4f})")
    if len(top100) >= 100:
        print(f"  Top 100: {top100[99]['cid']} (score={0.40 + ((top100[99]['final'] - mn) / rng) * 0.59:.4f})")
    print(f"  Honeypots removed: {hp_filtered}")
    print(f"{'=' * 60}")
    print(f"{'⚠️ OVER 5-MINUTE LIMIT!' if elapsed > 300 else '✅ Within time limit'}")


if __name__ == "__main__":
    main()
