"""
JobMatch AI — Resume vs JD Analyzer Backend
=============================================
NLP-based matching using TF-IDF + cosine similarity + LLM analysis.

Usage:
    python jobmatch.py
    curl -X POST http://localhost:8003/match -d '{"resume":"...","job_description":"..."}'

Requirements:
    pip install scikit-learn fastapi uvicorn spacy
    python -m spacy download en_core_web_sm
    pip install langchain langchain-huggingface huggingface-hub
"""

import os
import re
import json
import math
from collections import Counter
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# sklearn TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# HuggingFace LLM for detailed analysis
from langchain_huggingface import HuggingFaceEndpoint

HF_TOKEN = os.getenv("HF_API_TOKEN", "")

# ── Common skills taxonomy ────────────────────────────────────────────────────
SKILLS_TAXONOMY = {
    "data_analysis":  ["sql", "python", "r", "pandas", "numpy", "excel", "tableau", "power bi",
                       "data analysis", "data analytics", "statistical analysis", "eda"],
    "machine_learning": ["machine learning", "deep learning", "scikit-learn", "tensorflow",
                          "pytorch", "xgboost", "random forest", "neural network", "nlp",
                          "llm", "langchain", "rag", "transformers", "shap"],
    "engineering":    ["sql", "etl", "pipeline", "azure", "aws", "databricks", "spark",
                       "airflow", "docker", "kubernetes", "api", "fastapi", "flask"],
    "visualization":  ["power bi", "tableau", "matplotlib", "seaborn", "plotly", "looker",
                       "dax", "d3.js", "grafana"],
    "soft_skills":    ["communication", "leadership", "collaboration", "agile", "scrum",
                       "project management", "stakeholder", "presentation", "problem solving"],
    "cloud":          ["azure", "aws", "gcp", "databricks", "snowflake", "redshift",
                       "bigquery", "s3", "lambda", "ec2"],
}

ALL_SKILLS = list({s for skills in SKILLS_TAXONOMY.values() for s in skills})


# ─────────────────────────────────────────────────────────────────────────────
# NLP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-+#.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> List[str]:
    """Extract known skills from text using substring matching."""
    lower = preprocess(text)
    found = []
    for skill in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower):
            found.append(skill)
    return list(set(found))


def tfidf_similarity(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity between two documents."""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    try:
        tfidf = vectorizer.fit_transform([preprocess(text1), preprocess(text2)])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    except Exception:
        return 0.0


def keyword_overlap(text1: str, text2: str) -> float:
    """Compute keyword overlap ratio."""
    words1 = set(preprocess(text1).split())
    words2 = set(preprocess(text2).split())
    if not words2:
        return 0.0
    overlap = words1 & words2
    return len(overlap) / len(words2)


def score_match(resume: str, jd: str) -> dict:
    """
    Multi-signal matching:
    - TF-IDF cosine similarity          (40%)
    - Skill overlap                     (35%)
    - Keyword overlap                   (15%)
    - Experience/education keywords     (10%)
    """
    resume_skills = extract_skills(resume)
    jd_skills     = extract_skills(jd)

    if jd_skills:
        matched  = [s for s in resume_skills if s in jd_skills]
        missing  = [s for s in jd_skills if s not in resume_skills]
        bonus    = [s for s in resume_skills if s not in jd_skills]
        skill_score = len(matched) / len(jd_skills)
    else:
        matched, missing, bonus = resume_skills, [], []
        skill_score = 0.5

    tfidf_score   = tfidf_similarity(resume, jd)
    keyword_score = keyword_overlap(resume, jd)

    edu_keywords  = ["bachelor", "master", "m.s.", "b.s.", "phd", "degree", "university", "college"]
    exp_keywords  = ["years", "experience", "worked", "developed", "built", "managed", "led"]
    edu_score     = sum(1 for k in edu_keywords if k in resume.lower()) / len(edu_keywords)
    exp_score     = sum(1 for k in exp_keywords if k in resume.lower()) / len(exp_keywords)
    edu_exp_score = (edu_score + exp_score) / 2

    overall = (
        tfidf_score   * 0.40 +
        skill_score   * 0.35 +
        keyword_score * 0.15 +
        edu_exp_score * 0.10
    )
    overall = min(1.0, max(0.0, overall))
    score   = round(overall * 100)

    # Dimension scores for radar chart
    radar = {
        "technical":  min(100, round(skill_score * 100)),
        "experience": min(100, round(exp_score * 100)),
        "education":  min(100, round(edu_score * 100)),
        "keywords":   min(100, round(keyword_score * 100)),
        "culture":    min(100, round(tfidf_score * 100)),
    }

    return {
        "score":          score,
        "tfidf_sim":      round(tfidf_score, 3),
        "skill_overlap":  round(skill_score, 3),
        "matched_skills": matched[:10],
        "missing_skills": missing[:8],
        "bonus_skills":   bonus[:8],
        "radar":          radar,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_analysis(resume: str, jd: str, match_data: dict) -> str:
    """Get detailed LLM analysis and recommendations."""
    if not HF_TOKEN:
        # Fallback to rule-based analysis when no API token
        score = match_data["score"]
        matched = match_data["matched_skills"]
        missing = match_data["missing_skills"]
        verdict = "strong" if score >= 75 else "moderate" if score >= 55 else "weak"
        return (
            f"This is a {verdict} match at {score}%. "
            f"Your strongest matching skills are: {', '.join(matched[:4])}. "
            + (f"Key gaps to address: {', '.join(missing[:3])}. " if missing else "No critical skill gaps. ")
            + ("Consider tailoring your resume summary to mirror the job description language more closely." if score < 75 else "Apply immediately — your profile is well-aligned.")
        )

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=300,
        temperature=0.15,
        task="text-generation",
    )

    prompt = f"""You are an expert ATS recruiter. Analyze this resume-job match in 3 sentences.

Match score: {match_data['score']}%
Matched skills: {', '.join(match_data['matched_skills'][:6])}
Missing skills: {', '.join(match_data['missing_skills'][:4])}

Resume excerpt: {resume[:800]}

Job description excerpt: {jd[:600]}

In exactly 3 sentences: (1) overall verdict with score reasoning, (2) top 2 strengths specific to this JD, (3) top recommendation to improve the application.
Analysis:"""

    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        return f"AI analysis unavailable: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="JobMatch AI API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MatchRequest(BaseModel):
    resume:          str
    job_description: str
    use_llm:         bool = True


@app.post("/match")
async def match(req: MatchRequest):
    """Analyze resume vs job description match."""
    if len(req.resume.strip()) < 50:
        raise HTTPException(400, "Resume text is too short (min 50 chars).")
    if len(req.job_description.strip()) < 50:
        raise HTTPException(400, "Job description is too short (min 50 chars).")

    match_data = score_match(req.resume, req.job_description)

    score = match_data["score"]
    if score >= 80:
        headline = "Strong Match"
        sub      = "Your profile aligns well — apply with confidence."
    elif score >= 65:
        headline = "Good Match"
        sub      = "Solid fit with a few gaps to address."
    elif score >= 50:
        headline = "Moderate Match"
        sub      = "Relevant background but tailoring needed."
    else:
        headline = "Weak Match"
        sub      = "Significant gaps — consider upskilling first."

    analysis = ""
    if req.use_llm:
        analysis = get_llm_analysis(req.resume, req.job_description, match_data)

    return {
        "score":          score,
        "headline":       headline,
        "sub":            sub,
        "matched":        match_data["matched_skills"],
        "missing":        match_data["missing_skills"],
        "bonus":          match_data["bonus_skills"],
        "radar":          match_data["radar"],
        "analysis":       analysis,
        "debug": {
            "tfidf_sim":     match_data["tfidf_sim"],
            "skill_overlap": match_data["skill_overlap"],
        },
    }


@app.get("/skills")
async def list_skills():
    """Return the skills taxonomy used for matching."""
    return SKILLS_TAXONOMY


@app.get("/health")
async def health():
    return {"status": "ok", "llm_enabled": bool(HF_TOKEN)}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
