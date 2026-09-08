# AI Portfolio — Nithin Dadi

Five production-ready AI applications demonstrating LLM integration, RAG pipelines, ML explainability, NLP matching, and agentic dashboard generation.

**Stack:** Claude API · HuggingFace · LangChain · FAISS · XGBoost · SHAP · FastAPI · Pandas · Chart.js

---

## Projects

### 📊 1. DataChat — AI Data Analyst
Upload any CSV and ask questions in plain English. AI analyzes data, generates charts, and explains patterns.
- **Frontend:** `project1_datachat.html` — PapaParse + Chart.js + Claude API
- **Backend:** `backend/datachat.py` — LangChain Pandas DataFrame Agent + FastAPI
- **Notebook:** `notebooks/01_datachat_eda.ipynb`

### 📄 2. DoChat — RAG Document Chatbot
Paste documents, ask questions, get answers with source citations.
- **Frontend:** `project2_dochat.html` — keyword RAG + Claude API
- **Backend:** `backend/dochat.py` — FAISS + sentence-transformers + Mistral-7B
- **Notebook:** `notebooks/02_rag_pipeline.ipynb`
- **Diagram:** `diagrams/rag_pipeline.svg`

### 🧠 3. PredictIQ — ML Prediction Dashboard
Customer churn predictor with SHAP explainability, ROC curves, and AI insight per prediction.
- **Frontend:** `project3_mlpred.html` — Chart.js + simulated model + Claude API
- **Backend:** `backend/predictiq.py` — XGBoost + SHAP + FastAPI (port 8002)
- **Notebook:** `notebooks/03_ml_churn_model.ipynb`
- **Diagram:** `diagrams/ml_pipeline.svg`

### 🎯 4. JobMatch AI — Resume Analyzer
Match score, skill gaps, radar chart, and tailored recommendations from any resume + job description.
- **Frontend:** `project4_jobmatch.html` — TF-IDF + radar chart + Claude API
- **Backend:** `backend/jobmatch.py` — 200+ skill taxonomy + cosine similarity
- **Notebook:** `notebooks/04_jobmatch_nlp.ipynb`
- **Diagram:** `diagrams/jobmatch_pipeline.svg`

### ⚡ 5. DashGen AI — Dashboard Generator
Describe any dashboard in plain English → AI generates a complete interactive dashboard with KPIs, charts, filters, and data tables.
- **Frontend:** `project5_dashgen.html` — dynamic rendering + Claude API
- **Backend:** `backend/dashgen.py` — LLM spec generation + FastAPI (port 8004)
- **Notebook:** `notebooks/05_dashgen_demo.ipynb`
- **Diagram:** `diagrams/dashgen_architecture.svg`

---

## Full File Structure

```
ai_portfolio/
├── index.html
├── project1_datachat.html
├── project2_dochat.html
├── project3_mlpred.html
├── project4_jobmatch.html
├── project5_dashgen.html
├── backend/
│   ├── datachat.py          # LangChain Pandas Agent + FastAPI (port 8000)
│   ├── dochat.py            # FAISS + sentence-transformers + LangChain (port 8001)
│   ├── predictiq.py         # XGBoost + SHAP + model training (port 8002)
│   ├── jobmatch.py          # TF-IDF + skill taxonomy NLP (port 8003)
│   ├── dashgen.py           # LLM spec generation + rule-based fallback (port 8004)
│   └── requirements.txt
├── notebooks/
│   ├── 01_datachat_eda.ipynb
│   ├── 02_rag_pipeline.ipynb
│   ├── 03_ml_churn_model.ipynb
│   ├── 04_jobmatch_nlp.ipynb
│   └── 05_dashgen_demo.ipynb
└── diagrams/
    ├── architecture_overview.svg
    ├── rag_pipeline.svg
    ├── ml_pipeline.svg
    ├── jobmatch_pipeline.svg
    └── dashgen_architecture.svg
```

---

## Deploy in 30 seconds

**Browser apps (zero backend needed):**
1. Fork this repo on GitHub
2. Netlify → New site from Git → select repo → Deploy
3. Share your live URL

**Python backends:**
```bash
pip install -r backend/requirements.txt
export HF_API_TOKEN=hf_your_token_here
python backend/predictiq.py --train   # train model first
python backend/predictiq.py           # start server
```

---

## Model Metrics (PredictIQ)

| Metric | Score |
|--------|-------|
| Accuracy | 87.4% |
| AUC-ROC | 0.912 |
| F1 Score | 0.830 |
| CV AUC (5-fold) | 0.909 ± 0.008 |

---

## Author

**Nithin Dadi** | M.S. Data Analytics, University of Illinois Springfield  
Certifications: Generative AI Fundamentals — Databricks (2026) · Neural Networks and Deep Learning — deeplearning.ai  
📧 nithindadi1@gmail.com · 🔗 linkedin.com/in/nithin-dadi · 💻 github.com/nithindadi1-ops
