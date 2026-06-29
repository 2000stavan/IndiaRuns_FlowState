"""
reasoning.py — Template-based reasoning generation from verified JSON fields.

No model. Zero hallucination risk. Every fact comes directly from the candidate's
JSON profile. Stage 4 checks that every mentioned fact exists in the actual data.

Functions exported:
    generate_reasoning(candidate, rank, score) -> str

Imported by: rank.py (called AFTER final ranking to ensure correct rank-dependent tone)
"""

from datetime import date

# Skills to highlight when found (display names, case-sensitive for output)
JD_SKILLS_DISPLAY = {
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "Elasticsearch",
    "OpenSearch", "Sentence Transformers", "NLP", "PyTorch",
    "Hugging Face Transformers", "Recommendation Systems", "LoRA", "QLoRA",
    "PEFT", "Fine-tuning LLMs", "Python", "BM25", "MLflow",
    "Weights & Biases", "Embeddings", "Information Retrieval", "Ranking",
    "XGBoost", "LightGBM", "MLOps",
}

# Consulting firm names for concern flagging
CONS_CHECK = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "mindtree", "hexaware",
}

# Engineering role keywords for identifying product-company roles
_ENG_KW = {"engineer", "developer", "scientist", "architect", "lead"}


def generate_reasoning(candidate: dict, rank: int, score: float) -> str:
    """
    Generate a reasoning string for a ranked candidate.

    Every fact in the output is sourced directly from the candidate's JSON.
    The tone adapts to the rank:
        - Ranks 1-10:  Confident, at most one concern
        - Ranks 11-30: Moderate, up to two concerns
        - Ranks 31-70: Balanced, up to three concerns
        - Ranks 71-100: Lists multiple concerns

    Args:
        candidate: Full candidate dict from the dataset
        rank: Final rank (1-100) after all scoring and re-ranking
        score: Final normalized score (0.40-0.99)

    Returns:
        Reasoning string suitable for the CSV output
    """
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    strengths = []
    concerns = []

    # =========================================================================
    # STRENGTHS — only from verified JSON fields
    # =========================================================================

    # Years of experience and current title
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "Unknown")
    strengths.append(f"{yoe}y exp as {title}")

    # JD-relevant skills with trust signals
    matched = [
        sk["name"] for sk in skills
        if sk["name"] in JD_SKILLS_DISPLAY
        and sk.get("proficiency") in ("intermediate", "advanced", "expert")
        and sk.get("duration_months", 0) > 6
    ]
    if matched:
        strengths.append(f"hands-on: {', '.join(matched[:3])}")

    # Product company experience (non-consulting, non-tiny, engineering title)
    prod = [
        j for j in career
        if not any(f in j["company"].lower() for f in CONS_CHECK)
        and j["company_size"] not in ("1-10",)
        and any(e in j["title"].lower() for e in _ENG_KW)
    ]
    if prod:
        strengths.append(f"product co ({prod[0]['company']})")

    # Location
    location = p.get("location", "Unknown")
    strengths.append(f"based in {location.split(',')[0].strip()}")

    # GitHub activity (only if positive signal)
    gh = s.get("github_activity_score", -1)
    if gh >= 30:
        strengths.append(f"active GitHub ({int(gh)})")

    # Actively applying
    if s.get("applications_submitted_30d", 0) >= 2:
        strengths.append("actively applying")

    # Platform skill assessments (verified scores)
    assessments = s.get("skill_assessment_scores", {})
    jd_assessments = {
        k: v for k, v in assessments.items()
        if any(r in k.lower() for r in {
            "faiss", "pinecone", "embedding", "nlp", "retrieval",
            "ranking", "python", "mlflow", "elasticsearch",
        })
    }
    if jd_assessments:
        top_assessment = max(jd_assessments.items(), key=lambda x: x[1])
        if top_assessment[1] >= 60:
            strengths.append(f"verified {top_assessment[0]} ({top_assessment[1]:.0f}/100)")

    # =========================================================================
    # CONCERNS — only from verified JSON fields
    # =========================================================================

    # Inactivity
    try:
        days = (date.today() - date.fromisoformat(s["last_active_date"])).days
        if days > 90:
            concerns.append(f"inactive {days}d")
    except (ValueError, KeyError):
        pass

    # Low response rate
    rr = s.get("recruiter_response_rate", 1.0)
    if rr < 0.25:
        concerns.append(f"low response rate ({rr:.0%})")

    # Long notice period
    notice = s.get("notice_period_days", 0)
    if notice > 60:
        concerns.append(f"{notice}-day notice")

    # Not open to work
    if not s.get("open_to_work_flag", True):
        concerns.append("not open to work")

    # Low interview completion
    icr = s.get("interview_completion_rate", 1.0)
    if icr < 0.50:
        concerns.append(f"low interview completion ({icr:.0%})")

    # Location concern (outside India, not relocating)
    country = p.get("country", "India")
    if country.lower() not in ("india",) and not s.get("willing_to_relocate", False):
        concerns.append(f"in {country}, not relocating")

    # Entire career in consulting/IT-services
    if career and all(
        any(f in j["company"].lower() for f in CONS_CHECK)
        for j in career
    ):
        concerns.append("entire career in IT services")

    # Low profile completeness
    pc = s.get("profile_completeness_score", 100)
    if pc < 50:
        concerns.append(f"incomplete profile ({pc:.0f}%)")

    # =========================================================================
    # COMPOSE — rank determines tone
    # =========================================================================
    st = "; ".join(strengths)

    if rank <= 10:
        # Top 10: confident tone, at most one concern mentioned as a "note"
        if concerns:
            return f"{st}. Note: {concerns[0]}."
        else:
            return f"{st}; strong availability signals."

    elif rank <= 30:
        # Ranks 11-30: moderate, up to two concerns
        c = ", ".join(concerns[:2])
        return f"{st}. Concerns: {c}." if concerns else f"{st}."

    elif rank <= 70:
        # Ranks 31-70: balanced, up to three concerns
        c = ", ".join(concerns[:3])
        return f"{st}. Concerns: {c}." if concerns else f"{st}."

    else:
        # Ranks 71-100: list multiple concerns
        c = "; ".join(concerns)
        if concerns:
            return f"{st}. Multiple concerns: {c}."
        else:
            return f"{st}; included on partial skill fit."
