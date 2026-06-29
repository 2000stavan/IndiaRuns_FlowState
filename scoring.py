"""
scoring.py — All deterministic scoring functions for candidate ranking.

Imported by: precompute.py and rank.py
No model loading — pure Python logic only.

Functions exported:
    detect_honeypot(candidate) -> bool
    compute_career_arc_score(candidate) -> float
    compute_skill_trust_score(candidate) -> float
    compute_location_score(candidate) -> float
    compute_yoe_score(candidate) -> float
    compute_behavioral_score(signals) -> float
"""

from datetime import date as dt

# =============================================================================
# Constants
# =============================================================================

CONSULTING = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "hcl technologies", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "mindtree", "igate", "syntel", "mastech",
    "niit technologies", "patni", "satyam",
}

RESEARCH_KW = {
    "research scientist", "research engineer", "phd student", "phd candidate",
    "postdoc", "postdoctoral", "research intern", "research associate",
}

CV_SPEECH = {
    "computer vision", "object detection", "image classification", "yolo",
    "opencv", "speech recognition", "asr", "tts", "robotics",
}

NLP_IR = {
    "nlp", "information retrieval", "semantic search", "vector search",
    "embedding", "embeddings", "retrieval", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "elasticsearch", "opensearch", "recommendation",
    "ranking", "search", "bm25", "dense retrieval", "hybrid search",
    "sentence-transformers",
}

ENG_KW = {"engineer", "developer", "architect", "scientist", "lead", "sde", "swe"}

NON_ENG = {
    "hr manager", "human resources", "operations manager", "marketing manager",
    "content writer", "graphic designer", "accountant", "sales manager",
    "customer support", "civil engineer", "mechanical engineer",
}

# Title-chaser seniority-level keywords (ordered low → high)
TITLE_LEVELS = [
    "junior", "associate", "mid", "senior", "staff", "principal",
    "lead", "director", "vp", "vice president", "head",
]

# LangChain-only detection keywords
LANGCHAIN_KW = {
    "langchain", "openai api", "chatgpt", "gpt-4", "llm api", "api wrapper",
}

PRE_LLM_KW = {
    "sklearn", "xgboost", "lightgbm", "tensorflow", "pytorch", "keras",
    "spark", "kafka", "airflow", "recommendation", "ranking", "retrieval",
    "embedding", "neural network", "random forest", "gradient boosting",
}

# JD-relevant skills for trust scoring
JD_REQ = {
    "embeddings", "embedding", "sentence-transformers", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "vector database", "hybrid search", "dense retrieval",
    "information retrieval", "semantic search", "python", "nlp",
    "natural language processing", "ndcg", "mrr", "bm25",
    "recommendation", "ranking", "retrieval",
}

JD_NICE = {
    "lora", "qlora", "peft", "fine-tuning llms", "learning to rank",
    "xgboost", "lightgbm", "pytorch", "hugging face", "transformers",
    "recommendation systems", "mlflow",
}

JD_NEG = {
    "computer vision", "opencv", "yolo", "image classification",
    "object detection", "speech recognition", "asr", "tts", "robotics",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.20,
    "intermediate": 0.50,
    "advanced": 0.80,
    "expert": 1.00,
}


# =============================================================================
# Honeypot Detection
# =============================================================================

def detect_honeypot(candidate: dict) -> bool:
    """
    Detect impossible/fabricated candidate profiles.
    Returns True if the candidate is a honeypot (should be excluded).
    """
    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate["skills"]
    today = dt.today()

    # 1. Expert/advanced proficiency with zero duration on 3+ skills
    impossible = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 1) == 0
    )
    if impossible >= 3:
        return True

    # 2. Career months vs claimed YoE
    total = sum(j.get("duration_months", 0) for j in career)
    claimed = profile["years_of_experience"] * 12
    if total > claimed * 1.5 + 24:
        return True  # impossible overlap
    if total < claimed * 0.35 and len(career) > 1:
        return True  # impossible gap

    # 3. Future dates or end before start
    for job in career:
        try:
            start = dt.fromisoformat(job["start_date"])
            if start > today:
                return True
            if job["end_date"] is not None:
                end = dt.fromisoformat(job["end_date"])
                if end > today and not job.get("is_current", False):
                    return True
                if end < start:
                    return True
        except (ValueError, TypeError):
            pass

    # 4. Stated duration_months contradicts actual date range by >18 months
    for job in career:
        try:
            start = dt.fromisoformat(job["start_date"])
            end = (
                dt.fromisoformat(job["end_date"])
                if job["end_date"]
                else today if job.get("is_current") else None
            )
            if end is None:
                continue
            actual = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(actual - job.get("duration_months", 0)) > 18:
                return True
        except (ValueError, TypeError):
            pass

    # 5. Multiple skills with duration far exceeding total career length
    # Buffer of 48 months accounts for pre-career learning (college, side projects).
    # Require 3+ impossible skills — single outliers are common in real data.
    impossible_skills = sum(
        1 for sk in skills
        if sk.get("duration_months", 0) > total + 48
    )
    if impossible_skills >= 3:
        return True

    return False


# =============================================================================
# Career Arc Score (weight 0.25)
# =============================================================================

def _is_consulting(company_name: str) -> bool:
    """Check if a company name matches a known consulting/IT-services firm."""
    lower = company_name.lower()
    return any(f in lower for f in CONSULTING)


def compute_career_arc_score(candidate: dict) -> float:
    """
    Score based on career trajectory: product-company experience,
    engineering roles, relevant domain. Catches keyword stuffers,
    consulting-only, pure research, wrong domain.

    Returns 0.0-1.0 (higher = better fit).
    """
    career = candidate["career_history"]
    profile = candidate["profile"]
    skills = candidate["skills"]

    companies = [j["company"].lower() for j in career]
    titles = [j["title"].lower() for j in career]
    skill_names = {s["name"].lower() for s in skills}
    career_text = " ".join(j.get("description", "") for j in career).lower()

    # --- Hard disqualifiers (return immediately) ---

    # All consulting firms
    if all(_is_consulting(co) for co in companies):
        return 0.10

    # >75% consulting
    if len(companies) > 0:
        consulting_ratio = sum(1 for co in companies if _is_consulting(co)) / len(companies)
        if consulting_ratio > 0.75:
            return 0.25

    # Non-engineer keyword stuffer (HR Manager, Content Writer, etc.)
    curr = profile["current_title"].lower()
    if any(ne in curr for ne in NON_ENG):
        # Only allow if they have prior engineering titles
        if not any(any(e in t for e in ENG_KW) for t in titles[1:]):
            return 0.05  # keyword stuffer

    # Title-chaser: avg tenure < 18mo AND titles show seniority-level inflation
    # (NOT just unique titles — that falsely flags lateral specialization moves)
    if len(career) >= 3:
        tenures = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure < 18:
            # Check for actual seniority-level keyword inflation
            levels = []
            for j in career:
                t = j["title"].lower()
                for i, lvl in enumerate(TITLE_LEVELS):
                    if lvl in t:
                        levels.append(i)
                        break
            # Only flag if 2+ level keywords found AND they're strictly ascending
            if (len(levels) >= 2
                    and levels == sorted(levels)
                    and levels[0] != levels[-1]):
                return 0.20  # true title-chaser: climbing seniority ladder fast

    # LangChain-only: API-wrapper experience with no pre-LLM production ML
    has_langchain = any(k in career_text for k in LANGCHAIN_KW)
    has_pre_llm = any(k in career_text for k in PRE_LLM_KW)
    if has_langchain and not has_pre_llm and profile["years_of_experience"] < 4:
        return 0.15  # LangChain-only, no production ML foundation

    # Pure research (all titles are research, zero engineering)
    r_ct = sum(1 for t in titles if any(r in t for r in RESEARCH_KW))
    e_ct = sum(1 for t in titles if any(e in t for e in ENG_KW))
    if r_ct > 0 and e_ct == 0:
        return 0.15

    # CV/Speech primary without NLP/IR
    has_nlp = any(k in career_text for k in NLP_IR)
    cv_ct = sum(1 for s in skill_names if any(c in s for c in CV_SPEECH))
    nlp_ct = sum(1 for s in skill_names if any(n in s for n in NLP_IR))
    if cv_ct >= 3 and nlp_ct == 0 and not has_nlp:
        return 0.25

    # Senior non-IC role (Director, VP, etc.) without recent hands-on
    if any(m in curr for m in ["director", "vp ", "vice president", "head of", " cto", " cpo"]):
        recent = " ".join(j.get("description", "") for j in career[:2]).lower()
        if not any(w in recent for w in ["built", "implemented", "deployed", "wrote", "shipped"]):
            return 0.35

    # --- Positive signals ---
    score = 0.65

    # Product company engineering roles
    prod = [
        j for j in career
        if not _is_consulting(j["company"].lower())
        and j["company_size"] not in ("1-10",)
        and any(e in j["title"].lower() for e in ENG_KW)
    ]
    if len(prod) >= 2:
        score += 0.20
    elif len(prod) == 1:
        score += 0.12

    # NLP/IR experience in career descriptions
    if has_nlp:
        score += 0.10

    # Mostly engineering titles
    if e_ct / max(len(titles), 1) >= 0.75:
        score += 0.05

    return min(score, 1.0)


# =============================================================================
# Skill Trust Score (weight 0.15)
# =============================================================================

def compute_skill_trust_score(candidate: dict) -> float:
    """
    Score based on JD-relevant skills, weighted by proficiency,
    endorsements, and duration. Penalizes CV/speech-only skills.
    Includes bonus for platform-verified skill assessments.

    Returns 0.0-1.0 (higher = better fit).
    """
    req = 0.0
    nice = 0.0
    penalty = 0.0

    for sk in candidate["skills"]:
        name = sk["name"].lower()
        pw = PROFICIENCY_WEIGHT.get(sk["proficiency"], 0.5)
        end = min(sk.get("endorsements", 0), 50) / 50.0
        dur = min(sk.get("duration_months", 0), 36) / 36.0
        trust = 0.50 + end * 0.25 + dur * 0.25  # higher endorsements + duration = more trusted
        val = pw * trust

        if any(r in name for r in JD_REQ):
            req += val
        elif any(n in name for n in JD_NICE):
            nice += val * 0.5

        if any(g in name for g in JD_NEG):
            penalty += 0.04

    base_result = max(0.0, min(min(req / 3.0, 1.0) + min(nice / 2.0, 0.25) - penalty, 1.0))

    # Bonus: platform-verified skill assessment scores
    assessment_bonus = 0.0
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for skill_name, score in assessment_scores.items():
        if any(r in skill_name.lower() for r in JD_REQ):
            # Score is 0-100; normalize and apply small bonus
            assessment_bonus += (score / 100.0) * 0.05  # up to 0.05 per assessed skill

    assessment_bonus = min(assessment_bonus, 0.10)  # cap total bonus at 0.10

    return max(0.0, min(base_result + assessment_bonus, 1.0))


# =============================================================================
# Location Score (weight 0.10)
# =============================================================================

def compute_location_score(candidate: dict) -> float:
    """
    Score based on candidate location relative to JD requirements.
    Pune/Noida = ideal. Tier-1 Indian cities = good. Outside India = penalty.

    Returns 0.0-1.0 (higher = better fit).
    """
    loc = candidate["profile"]["location"].lower()
    country = candidate["profile"].get("country", "").lower()
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)

    # Tier A: Primary locations
    if any(c in loc for c in ["pune", "noida"]):
        return 1.00

    # Tier B: Major Indian cities
    if any(c in loc for c in [
        "hyderabad", "bangalore", "bengaluru", "delhi",
        "gurgaon", "gurugram", "mumbai", "ncr", "new delhi",
    ]):
        return 0.85

    # Tier C/D: Other Indian cities
    if country == "india" or "india" in loc:
        if any(c in loc for c in [
            "chennai", "kolkata", "ahmedabad", "kochi", "jaipur",
            "trivandrum", "chandigarh", "indore", "nagpur",
        ]):
            return 0.70
        return 0.65  # India, city unknown or unlisted

    # Tier E/F: Outside India
    return 0.45 if relocate else 0.15


# =============================================================================
# Years of Experience Score (weight 0.10)
# =============================================================================

def compute_yoe_score(candidate: dict) -> float:
    """
    Score based on years of experience relative to JD sweet spot (5-9 years).

    Returns 0.0-1.0 (higher = better fit).
    """
    yoe = candidate["profile"]["years_of_experience"]

    if 6 <= yoe <= 8:
        return 1.00   # JD sweet spot
    if 5 <= yoe < 6:
        return 0.90
    if 8 < yoe <= 10:
        return 0.85
    if 4 <= yoe < 5:
        return 0.70
    if 10 < yoe <= 15:
        return 0.70   # potentially overqualified
    if yoe > 15:
        return 0.50
    return 0.40       # under 4 years — too junior


# =============================================================================
# Behavioral Score (multiplier, not a component of skill_fit)
# =============================================================================

def compute_behavioral_score(signals: dict) -> float:
    """
    Behavioral engagement score based on platform activity signals.
    Applied as a MULTIPLIER on skill_fit_composite, not as an additive component.

    Returns 0.0-1.0 (higher = more engaged/available).
    """
    today = dt.today()

    # Recency (30% weight) — most important
    try:
        days = (today - dt.fromisoformat(signals["last_active_date"])).days
    except (ValueError, KeyError):
        days = 365
    if days <= 30:
        recency = 1.00
    elif days <= 60:
        recency = 0.90
    elif days <= 90:
        recency = 0.75
    elif days <= 180:
        recency = 0.50
    else:
        recency = 0.20  # ghost

    # Response rate (25% weight)
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr >= 0.60:
        response = 1.00
    elif rr >= 0.40:
        response = 0.85
    elif rr >= 0.20:
        response = 0.65
    else:
        response = 0.30  # 5% = ghost

    # Notice period (20% weight)
    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        notice_s = 1.00
    elif notice <= 60:
        notice_s = 0.85
    elif notice <= 90:
        notice_s = 0.65
    else:
        notice_s = 0.40

    # Open to work (10% weight)
    open_work = 1.0 if signals.get("open_to_work_flag", True) else 0.75

    # Interview completion (10% weight)
    icr = signals.get("interview_completion_rate", 0.75)
    interview = 0.50 + (icr * 0.50)  # 0.0→0.50, 1.0→1.00

    # GitHub activity (3% weight — reduced from 5% to make room for saved_by_recruiters)
    gh = signals.get("github_activity_score", -1)
    if gh == -1:
        github = 0.85   # -1 = no GitHub = NEUTRAL
    elif gh >= 50:
        github = 1.00
    elif gh >= 20:
        github = 0.90
    else:
        github = 0.75

    # Saved by recruiters (2% weight — platform-validated demand signal)
    saved = min(signals.get("saved_by_recruiters_30d", 0), 20) / 20.0

    return (
        recency * 0.30
        + response * 0.25
        + notice_s * 0.20
        + open_work * 0.10
        + interview * 0.10
        + github * 0.03
        + saved * 0.02
    )
