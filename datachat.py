"""
DataChat — AI Data Analyst Backend
====================================
LangChain + HuggingFace + Pandas DataFrame Agent
Supports natural language queries over any CSV dataset.

Usage:
    python datachat.py --csv your_data.csv
    python datachat.py --csv your_data.csv --port 8000

Requirements:
    pip install langchain langchain-community langchain-experimental
    pip install huggingface-hub sentence-transformers
    pip install pandas matplotlib seaborn fastapi uvicorn python-multipart
"""

import os
import io
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import Optional

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import AgentAction, AgentFinish

# ── Config ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_API_TOKEN", "")          # export HF_API_TOKEN=hf_xxx
MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"  # free on HuggingFace Hub

# ── Global state (simple in-memory, replace with Redis for prod) ──────────────
_df: Optional[pd.DataFrame] = None
_agent = None

app = FastAPI(title="DataChat API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_llm() -> HuggingFaceEndpoint:
    """Instantiate a free HuggingFace Inference Endpoint LLM."""
    return HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.1,
        task="text-generation",
    )


def build_agent(df: pd.DataFrame):
    """Create a LangChain Pandas DataFrame agent."""
    llm = build_llm()
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,  # required for exec
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def df_summary(df: pd.DataFrame) -> dict:
    """Return a structured summary of the dataframe."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(include="object").columns.tolist()

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }

    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        summary["numeric_stats"] = desc.to_dict()

    if text_cols:
        summary["top_values"] = {
            col: df[col].value_counts().head(5).to_dict()
            for col in text_cols[:5]
        }

    return summary


def generate_chart(df: pd.DataFrame, chart_type: str, x: str, y: Optional[str] = None) -> str:
    """Generate a matplotlib chart and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    PURPLE = "#6366f1"
    GREEN  = "#10b981"

    if chart_type == "histogram" and x in df.columns:
        df[x].dropna().plot(kind="hist", ax=ax, color=PURPLE, edgecolor="#333", bins=20)
        ax.set_xlabel(x, color="#94a3b8")

    elif chart_type == "bar" and x in df.columns:
        counts = df[x].value_counts().head(10)
        counts.plot(kind="bar", ax=ax, color=PURPLE)
        ax.set_xlabel(x, color="#94a3b8")

    elif chart_type == "scatter" and x in df.columns and y and y in df.columns:
        df.plot.scatter(x=x, y=y, ax=ax, color=PURPLE, alpha=0.6)

    elif chart_type == "line" and x in df.columns and y and y in df.columns:
        df.plot(x=x, y=y, ax=ax, color=PURPLE, linewidth=2)

    elif chart_type == "box" and x in df.columns:
        df[[x]].boxplot(ax=ax, color={"boxes": PURPLE, "whiskers": GREEN, "caps": GREEN, "medians": "#f59e0b"})

    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_color("#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file and initialise the agent."""
    global _df, _agent

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    content = await file.read()
    _df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    _agent = build_agent(_df)

    return {"status": "ok", "summary": df_summary(_df)}


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_data(req: QueryRequest):
    """Ask a natural language question about the uploaded dataset."""
    global _df, _agent

    if _df is None:
        raise HTTPException(400, "No dataset loaded. Upload a CSV first.")

    try:
        result = _agent.invoke(req.question)
        answer = result.get("output", str(result))
    except Exception as e:
        answer = f"Could not complete analysis: {e}"

    return {"answer": answer, "question": req.question}


class ChartRequest(BaseModel):
    chart_type: str          # histogram | bar | scatter | line | box
    x_column: str
    y_column: Optional[str] = None


@app.post("/chart")
async def create_chart(req: ChartRequest):
    """Generate a chart for the current dataset."""
    if _df is None:
        raise HTTPException(400, "No dataset loaded.")

    img_b64 = generate_chart(_df, req.chart_type, req.x_column, req.y_column)
    return {"image_base64": img_b64, "format": "png"}


@app.get("/summary")
async def get_summary():
    """Return the current dataset summary."""
    if _df is None:
        raise HTTPException(400, "No dataset loaded.")
    return df_summary(_df)


@app.get("/health")
async def health():
    return {"status": "ok", "dataset_loaded": _df is not None}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
