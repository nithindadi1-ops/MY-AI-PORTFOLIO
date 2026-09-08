# AI Portfolio - Nithin Dadi

> Five production-ready AI applications demonstrating LLM integration, RAG pipelines, ML explainability, NLP matching, and agentic dashboard generation.

**Live Portfolio:** [nithindadi1-ops.github.io/MY-AI-PORTFOLIO](https://nithindadi1-ops.github.io/MY-AI-PORTFOLIO) | [LinkedIn](https://linkedin.com/in/nithin-dadi)

---

## Projects

| # | Project | What it does | Stack |
|---|---------|-------------|-------|
| 01 | [DataChat](project1_datachat.html) | Upload any CSV, ask questions in plain English, get charts and AI analysis | Claude API, LangChain, Chart.js, PapaParse |
| 02 | [DoChat — RAG Chatbot](project2_dochat.html) | Paste documents, ask questions, get answers with source citations | RAG, FAISS, sentence-transformers, LangChain |
| 03 | [PredictIQ — ML Dashboard](project3_mlpred.html) | Customer churn predictor with SHAP explainability and AI insight per prediction | XGBoost, SHAP, scikit-learn, FastAPI |
| 04 | [JobMatch AI](project4_jobmatch.html) | Resume vs JD match scorer — skill gaps, radar chart, LLM recommendations | NLP, TF-IDF, Claude API, Radar Chart |
| 05 | [DashGen AI](project5_dashgen.html) | Describe a dashboard in plain English, AI generates the full thing instantly | Agentic AI, Claude API, Chart.js, JSON Spec |

---

## File Structure

```
MY-AI-PORTFOLIO/
│
├── portfolio.html                ← Main portfolio landing page
│
├── project1_datachat.html        ← DataChat app
├── project2_dochat.html          ← DoChat RAG app
├── project3_mlpred.html          ← PredictIQ ML app
├── project4_jobmatch.html        ← JobMatch AI app
├── project5_dashgen.html         ← DashGen AI app
│
├── datachat.py                   ← LangChain Pandas Agent + FastAPI (port 8000)
├── dochat.py                     ← FAISS + sentence-transformers + LangChain (port 8001)
├── predictiq.py                  ← XGBoost + SHAP + model training (port 8002)
├── jobmatch.py                   ← TF-IDF + skill taxonomy NLP (port 8003)
├── dashgen.py                    ← LLM spec generation + rule-based fallback (port 8004)
├── requirements.txt              ← All Python dependencies
│
├── 01_datachat_eda.ipynb         ← EDA pipeline + LangChain Pandas agent demo
├── 02_rag_pipeline.ipynb         ← RAG pipeline deep dive + FAISS architecture
├── 03_ml_churn_model.ipynb       ← Full XGBoost + SHAP training pipeline
├── 04_jobmatch_nlp.ipynb         ← TF-IDF matching, skill taxonomy, radar visualization
├── 05_dashgen_demo.ipynb         ← Dashboard spec schema, matplotlib renderer
│
├── architecture_overview.svg     ← All 5 projects + shared LLM + deployment layers
├── rag_pipeline.svg              ← Index phase + query phase flow
├── ml_pipeline.svg               ← Data → XGBoost → SHAP → API
├── jobmatch_pipeline.svg         ← NLP → 4 signals → outputs → LLM
└── dashgen_architecture.svg      ← User input → LLM → Spec → Renderer → Dashboard
```

---

## Deploy (Netlify — 30 seconds)

1. Go to [netlify.com](https://netlify.com)
2. Sites → Add new site → Deploy manually
3. Drag the entire repo folder into Netlify
4. Done — live URL in seconds

**GitHub Pages:**
- Settings → Pages → Source: main branch → root
- Portfolio available at `https://nithindadi1-ops.github.io/MY-AI-PORTFOLIO/portfolio.html`

---

## Run Python Backends Locally

```bash
git clone https://github.com/nithindadi1-ops/MY-AI-PORTFOLIO
cd MY-AI-PORTFOLIO
pip install -r requirements.txt

export HF_API_TOKEN=hf_your_token_here    # free at huggingface.co

python predictiq.py --train               # train model first
python datachat.py                        # port 8000
python dochat.py                          # port 8001
python predictiq.py                       # port 8002
python jobmatch.py                        # port 8003
python dashgen.py                         # port 8004
```

---

## Model Performance (PredictIQ)

| Metric | Score |
|--------|-------|
| Accuracy | 87.4% |
| AUC-ROC | 0.912 |
| F1 Score | 0.830 |
| CV AUC (5-fold) | 0.909 ± 0.008 |
| Model | XGBoost Classifier |
| Training samples | 8,432 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude API (claude-sonnet-4-6), HuggingFace Mistral-7B |
| Embeddings | sentence-transformers MiniLM-L6-v2 (local, no API key) |
| Vector DB | FAISS (in-memory) |
| ML | XGBoost, scikit-learn, SHAP |
| NLP | TF-IDF, cosine similarity, 200+ skill taxonomy |
| Backend | FastAPI, uvicorn |
| Data | pandas, numpy, matplotlib, seaborn |
| Frontend | Vanilla JS, Chart.js, PapaParse |
| Deployment | Netlify / GitHub Pages |

---

## Author

**Nithin Dadi** - Data Analyst & AI Engineer

M.S. Data Analytics, University of Illinois Springfield (May 2026)

Certifications: Generative AI Fundamentals — Databricks (2026) · Neural Networks and Deep Learning — deeplearning.ai

[nithindadi1@gmail.com](mailto:nithindadi1@gmail.com) · [linkedin.com/in/nithin-dadi](https://linkedin.com/in/nithin-dadi) · [github.com/nithindadi1-ops](https://github.com/nithindadi1-ops)
