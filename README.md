# Redrob Intelligent Candidate Ranking

A two-stage hybrid AI pipeline that ranks 100K candidates against a Senior AI Engineer job description — not by keyword matching, but by actually understanding who fits the role.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE (precompute.py)                       │
│                                                                 │
│  candidates.jsonl ──► Feature Extraction ──► features.pkl       │
│         │               (scoring.py)          cids.pkl          │
│         │                                                       │
│         └──► BGE Embedding ──► FAISS IndexFlatIP ──► index.bin  │
│              (bge-small-en-v1.5)                    jd.npy      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RUNTIME (rank.py)                            │
│                     ≤ 5 min, CPU, 16 GB, no network              │
│                                                                 │
│  1. FAISS search ──► top 3000 candidates                        │
│  2. Multi-signal scoring + honeypot filter ──► top 200          │
│  3. Cross-encoder re-rank (ms-marco-MiniLM-L-6-v2) ──► top 100 │
│  4. Reasoning generation (template-based) ──► submission.csv    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pre-compute (offline, ~3 hours for 100K)
```bash
python precompute.py --candidates data/candidates.jsonl
```
This generates:
- `embeddings/index.bin` — FAISS index (100K × 384 dimensions)
- `embeddings/jd.npy` — Job description embedding
- `cache/features.pkl` — Pre-computed scoring features
- `cache/cids.pkl` — Candidate ID list

### 3. Rank (runtime, <5 minutes)
```bash
python rank.py --candidates data/candidates.jsonl --out output/submission.csv
```

### 4. Validate
```bash
python validate_submission.py output/submission.csv
```

## Scoring Components

### Skill-Fit Composite (5 signals)

| Component | Weight | What it measures |
|---|---|---|
| **Semantic Similarity** | 40% | Cosine similarity between career descriptions and JD (BGE embeddings) |
| **Career Arc** | 25% | Product-company engineering experience, catches keyword stuffers, consulting-only, title-chasers |
| **Skill Trust** | 15% | JD-relevant skills weighted by proficiency × endorsements × duration + assessment bonus |
| **Location** | 10% | Pune/Noida = 1.0, Tier-1 Indian cities = 0.85, outside India = 0.15-0.45 |
| **Years of Experience** | 10% | Sweet spot 6-8y = 1.0, under 4y = 0.40, over 15y = 0.50 |

### Behavioral Multiplier (applied on skill-fit)

| Signal | Weight |
|---|---|
| Last active recency | 30% |
| Recruiter response rate | 25% |
| Notice period | 20% |
| Open to work flag | 10% |
| Interview completion rate | 10% |
| GitHub activity | 3% |
| Saved by recruiters (30d) | 2% |

### Cross-Encoder Re-ranking

Top 200 candidates from multi-signal scoring are re-ranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Final score blends 75% multi-signal + 25% cross-encoder.

## Trap Detection

### Honeypot Detection (`detect_honeypot()`)
- Expert/advanced proficiency with 0 duration on 3+ skills
- Career months vs claimed YoE (impossible overlap/gap)
- Future dates or end-before-start
- Duration contradicts date range by >18 months
- 3+ skill durations exceeding career + 48 months

### Career Arc Penalties
- **Keyword stuffers**: HR Manager, Content Writer, etc. with no engineering history → 0.05
- **Consulting-only**: All TCS/Infosys/Wipro/etc. → 0.10
- **Title-chasers**: Seniority-level inflation (junior→senior→director) with <18mo avg tenure → 0.20
- **LangChain-only**: API-wrapper experience, no pre-LLM production ML, <4y → 0.15
- **Wrong domain**: CV/Speech primary without NLP/IR → 0.25

## Reasoning

Template-based from verified JSON fields only — zero hallucination risk. Every fact (YoE, title, company, skills, location) is pulled directly from the candidate's data.

**Rank-dependent tone:**
- Ranks 1-10: Confident, at most one concern as a "note"
- Ranks 11-30: Moderate, up to two concerns
- Ranks 31-70: Balanced, up to three concerns
- Ranks 71-100: Lists multiple concerns

## Models Used

| Model | Purpose | Size | When |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | Bi-encoder embedding | 33 MB | Offline only (precompute.py) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranking | 85 MB | Runtime (rank.py), top 200 only |

Both models are downloaded to `~/.cache/huggingface` during first run (requires internet). All subsequent runs work fully offline.

## File Structure

```
├── precompute.py          # Offline: embeddings + features
├── rank.py                # Runtime: ranking pipeline
├── scoring.py             # All deterministic scoring functions
├── reasoning.py           # Template-based reasoning generation
├── requirements.txt       # Pinned dependencies
├── submission_metadata.yaml
├── validate_submission.py # Official submission validator
├── data/                  # Symlinks to challenge data
├── embeddings/            # FAISS index + JD embedding
├── cache/                 # Pre-computed features
└── output/                # Generated CSV submissions
```

## Design Decisions

1. **Embed career descriptions, not skill names** — Skill names are trivially gameable (keyword stuffing). Career descriptions contain verifiable context about what the candidate actually built.

2. **75/25 cross-encoder blend** — Testing showed 60/40 lets the passage-retrieval CE model override domain-specific multi-signal scoring. 75/25 keeps CE as a refinement signal.

3. **No LLM for reasoning** — Template-based reasoning from verified JSON fields eliminates hallucination risk entirely. Stage 4 checks every fact against actual data.

4. **Seniority-level title-chaser detection** — Naive "unique title count" falsely flags lateral specialization moves (e.g., NLP Engineer → Search Engineer → Recommendation Systems Engineer). Checking for actual seniority keywords (junior→senior→director) catches real title-chasers without false positives.

## Constraints Met

- ✅ Runtime ≤ 5 minutes on CPU, 16 GB RAM
- ✅ No network access during ranking
- ✅ No GPU required
- ✅ Top 100 candidates with rank, score, reasoning
- ✅ Scores in 0.40–0.99 range, 4 decimal places
- ✅ Tie-breaking by candidate_id ascending
