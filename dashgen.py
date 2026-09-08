"""
DashGen AI — Dashboard Generator Backend
==========================================
Takes a natural language dashboard description and returns a complete
dashboard specification as structured JSON. Optionally accepts CSV data
to generate dashboards from real datasets.

Usage:
    python dashgen.py
    curl -X POST http://localhost:8004/generate \
      -H "Content-Type: application/json" \
      -d '{"description": "Sales dashboard for Q3 2025", "theme": "#3b82f6"}'

Requirements:
    pip install fastapi uvicorn pandas numpy langchain langchain-huggingface
"""

import os
import json
import re
import io
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HF_TOKEN = os.getenv("HF_API_TOKEN", "")

app = FastAPI(title="DashGen AI API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class KPI(BaseModel):
    label:  str
    value:  str
    change: str
    trend:  str          # "up" | "down" | "flat"
    sub:    Optional[str] = ""

class Dataset(BaseModel):
    label: str
    data:  List[float]

class Chart(BaseModel):
    type:     str        # "bar" | "line" | "doughnut" | "pie"
    title:    str
    span:     int = 1   # 1 = half width, 2 = full width
    labels:   List[str]
    datasets: List[Dataset]

class Filter(BaseModel):
    label:   str
    options: List[str]

class Table(BaseModel):
    title:   str
    headers: List[str]
    rows:    List[List[str]]

class DashboardSpec(BaseModel):
    title:    str
    subtitle: str
    theme:    str
    kpis:     List[KPI]
    filters:  List[Filter]
    charts:   List[Chart]
    table:    Table


class GenerateRequest(BaseModel):
    description: str
    theme:       str = "#3b82f6"
    csv_context: Optional[str] = None   # pre-parsed CSV summary string


# ─────────────────────────────────────────────────────────────────────────────
# DATA PROFILING
# ─────────────────────────────────────────────────────────────────────────────

def profile_csv(df: pd.DataFrame) -> str:
    """Generate a compact summary of a dataframe for the LLM context."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    text    = df.select_dtypes(include="object").columns.tolist()

    lines = [
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Rows: {len(df):,}",
        f"Numeric columns: {', '.join(numeric)}",
        f"Categorical columns: {', '.join(text)}",
    ]

    if numeric:
        desc = df[numeric].describe().round(2)
        lines.append(f"Numeric stats:\n{desc.to_string()}")

    for col in text[:3]:
        top = df[col].value_counts().head(5).to_dict()
        lines.append(f"Top values in '{col}': {top}")

    lines.append(f"Sample rows:\n{df.head(3).to_string()}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM DASHBOARD GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(req: GenerateRequest) -> str:
    csv_ctx = f"\n\nCSV DATA CONTEXT:\n{req.csv_context}" if req.csv_context else ""

    return f"""You are DashGen AI, an expert business intelligence dashboard designer.
Generate a complete, realistic dashboard specification as JSON.

User request: "{req.description}"{csv_ctx}

Return ONLY valid JSON matching this schema exactly (no markdown, no explanation):
{{
  "title": "Dashboard title",
  "subtitle": "Period or description",
  "theme": "{req.theme}",
  "kpis": [
    {{"label": "Metric", "value": "$1.2M", "change": "+12%", "trend": "up", "sub": "vs last quarter"}}
  ],
  "filters": [
    {{"label": "Region", "options": ["All","North","South","East","West"]}}
  ],
  "charts": [
    {{
      "type": "bar",
      "title": "Revenue by Region",
      "span": 1,
      "labels": ["North","South","East","West"],
      "datasets": [{{"label": "Revenue ($K)", "data": [420,310,580,290]}}]
    }},
    {{
      "type": "line",
      "title": "Monthly Trend",
      "span": 2,
      "labels": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"],
      "datasets": [{{"label": "Value", "data": [110,125,118,142,138,155,148,170,165]}}]
    }},
    {{
      "type": "doughnut",
      "title": "Category Breakdown",
      "span": 1,
      "labels": ["Cat A","Cat B","Cat C","Cat D"],
      "datasets": [{{"label": "Share", "data": [35,28,22,15]}}]
    }}
  ],
  "table": {{
    "title": "Detail View",
    "headers": ["Name","Value","Change","Status"],
    "rows": [
      ["Item A","$420K","+8%","On Track"],
      ["Item B","$310K","-3%","At Risk"]
    ]
  }}
}}

Rules:
- Generate 4-6 KPIs with realistic, formatted values ($, %, K, M)
- Generate 3-4 charts mixing types (bar, line, doughnut). Use span=2 for trend/line charts.
- Generate 1-2 filters with 4-6 meaningful options
- Generate 5-8 table rows with realistic data
- All numbers must be internally consistent
- KPI trend must be exactly "up", "down", or "flat"
- Use domain-appropriate terminology matching the user's request"""


def parse_spec(raw: str) -> dict:
    """Extract and validate JSON from LLM response."""
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()

    # Find first { ... } block
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(clean[start:end])


def generate_spec_hf(req: GenerateRequest) -> dict:
    """Generate dashboard spec using HuggingFace LLM."""
    from langchain_huggingface import HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1500,
        temperature=0.15,
        task="text-generation",
    )
    raw = llm.invoke(build_prompt(req))
    return parse_spec(raw)


def generate_spec_rule_based(req: GenerateRequest) -> dict:
    """
    Rule-based fallback dashboard generator — used when no API token is set.
    Generates a realistic spec from keywords in the description.
    """
    desc_lower = req.description.lower()

    # Domain detection
    if any(w in desc_lower for w in ["health", "hospital", "patient", "clinical", "medical"]):
        return _healthcare_spec(req.theme)
    elif any(w in desc_lower for w in ["sales", "revenue", "ecommerce", "product", "customer"]):
        return _sales_spec(req.theme)
    elif any(w in desc_lower for w in ["hr", "workforce", "employee", "hiring", "attrition"]):
        return _hr_spec(req.theme)
    elif any(w in desc_lower for w in ["finance", "budget", "p&l", "ebitda", "cash"]):
        return _finance_spec(req.theme)
    else:
        return _ops_spec(req.theme)


def _healthcare_spec(theme: str) -> dict:
    return {
        "title": "Healthcare Operations Dashboard",
        "subtitle": "Q3 2025 | All Departments",
        "theme": theme,
        "kpis": [
            {"label": "Total Admissions", "value": "12,847", "change": "+8.2%", "trend": "up", "sub": "vs Q2 2025"},
            {"label": "Bed Occupancy Rate", "value": "84.3%", "change": "+2.1%", "trend": "up", "sub": "target: 85%"},
            {"label": "Avg Length of Stay", "value": "4.2 days", "change": "-0.3d", "trend": "up", "sub": "improving"},
            {"label": "Readmission Rate", "value": "11.2%", "change": "-1.4%", "trend": "up", "sub": "30-day rate"},
            {"label": "Patient Satisfaction", "value": "4.3/5.0", "change": "+0.2", "trend": "up", "sub": "HCAHPS score"},
        ],
        "filters": [
            {"label": "Department", "options": ["All", "Emergency", "ICU", "Surgery", "Cardiology", "Pediatrics"]},
            {"label": "Period", "options": ["Q3 2025", "Q2 2025", "Q1 2025", "FY 2024"]},
        ],
        "charts": [
            {"type": "bar", "title": "Admissions by Department", "span": 1,
             "labels": ["Emergency", "Surgery", "ICU", "Cardiology", "Pediatrics", "Oncology"],
             "datasets": [{"label": "Admissions", "data": [3820, 2410, 1890, 1650, 1740, 1337]}]},
            {"type": "line", "title": "Monthly Admissions Trend", "span": 2,
             "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
             "datasets": [{"label": "Admissions", "data": [1180, 1050, 1290, 1340, 1410, 1380, 1450, 1520, 1227]}]},
            {"type": "doughnut", "title": "Bed Utilization", "span": 1,
             "labels": ["Occupied", "Available", "Under Maintenance"],
             "datasets": [{"label": "Beds", "data": [843, 124, 33]}]},
        ],
        "table": {
            "title": "Department Performance Summary",
            "headers": ["Department", "Admissions", "Occupancy", "Avg LOS", "Satisfaction"],
            "rows": [
                ["Emergency", "3,820", "91.2%", "1.8d", "4.1"],
                ["Surgery", "2,410", "88.5%", "5.2d", "4.4"],
                ["ICU", "1,890", "94.1%", "6.8d", "4.2"],
                ["Cardiology", "1,650", "82.3%", "4.1d", "4.5"],
                ["Pediatrics", "1,740", "76.8%", "3.4d", "4.6"],
                ["Oncology", "1,337", "79.4%", "5.9d", "4.3"],
            ],
        },
    }


def _sales_spec(theme: str) -> dict:
    return {
        "title": "Sales Performance Dashboard",
        "subtitle": "Q3 2025 | All Regions",
        "theme": theme,
        "kpis": [
            {"label": "Total Revenue", "value": "$8.4M", "change": "+14.2%", "trend": "up", "sub": "vs Q3 2024"},
            {"label": "Orders", "value": "24,810", "change": "+9.1%", "trend": "up", "sub": "this quarter"},
            {"label": "Avg Order Value", "value": "$338", "change": "+4.7%", "trend": "up", "sub": "per order"},
            {"label": "Conversion Rate", "value": "3.82%", "change": "-0.12%", "trend": "down", "sub": "from 3.94%"},
            {"label": "Customer Retention", "value": "78.4%", "change": "+2.3%", "trend": "up", "sub": "90-day rate"},
        ],
        "filters": [
            {"label": "Region", "options": ["All Regions", "North", "South", "East", "West"]},
            {"label": "Product", "options": ["All Products", "Category A", "Category B", "Category C"]},
        ],
        "charts": [
            {"type": "bar", "title": "Revenue by Region", "span": 1,
             "labels": ["North", "South", "East", "West", "Central"],
             "datasets": [{"label": "Revenue ($K)", "data": [2100, 1850, 2420, 1380, 650]}]},
            {"type": "line", "title": "Monthly Revenue Trend", "span": 2,
             "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
             "datasets": [
                 {"label": "2025", "data": [780, 820, 910, 950, 1020, 1080, 1150, 1180, 510]},
                 {"label": "2024", "data": [680, 710, 790, 820, 870, 940, 980, 1020, 450]},
             ]},
            {"type": "doughnut", "title": "Revenue by Category", "span": 1,
             "labels": ["Category A", "Category B", "Category C", "Category D"],
             "datasets": [{"label": "Revenue", "data": [38, 27, 21, 14]}]},
        ],
        "table": {
            "title": "Top Products by Revenue",
            "headers": ["Product", "Revenue", "Units", "Growth", "Margin"],
            "rows": [
                ["Product Alpha", "$1.82M", "5,410", "+18.4%", "42%"],
                ["Product Beta", "$1.44M", "4,250", "+12.1%", "38%"],
                ["Product Gamma", "$1.21M", "3,580", "+8.7%", "45%"],
                ["Product Delta", "$0.98M", "2,890", "-2.3%", "31%"],
                ["Product Epsilon", "$0.84M", "2,480", "+21.5%", "51%"],
            ],
        },
    }


def _hr_spec(theme: str) -> dict:
    return {
        "title": "HR Workforce Analytics Dashboard",
        "subtitle": "FY 2025 | All Departments",
        "theme": theme,
        "kpis": [
            {"label": "Total Headcount", "value": "1,842", "change": "+47", "trend": "up", "sub": "active employees"},
            {"label": "Attrition Rate", "value": "12.4%", "change": "-1.8%", "trend": "up", "sub": "YTD improvement"},
            {"label": "Time to Hire", "value": "28 days", "change": "-4d", "trend": "up", "sub": "avg this quarter"},
            {"label": "Engagement Score", "value": "73/100", "change": "+5", "trend": "up", "sub": "pulse survey"},
            {"label": "Open Roles", "value": "84", "change": "+12", "trend": "down", "sub": "current openings"},
        ],
        "filters": [
            {"label": "Department", "options": ["All", "Engineering", "Sales", "Operations", "HR", "Finance"]},
            {"label": "Level", "options": ["All Levels", "Individual Contributor", "Manager", "Director", "VP+"]},
        ],
        "charts": [
            {"type": "bar", "title": "Headcount by Department", "span": 1,
             "labels": ["Engineering", "Sales", "Operations", "Finance", "HR", "Marketing"],
             "datasets": [{"label": "Employees", "data": [520, 380, 440, 180, 92, 230]}]},
            {"type": "line", "title": "Hiring vs Attrition (Monthly)", "span": 2,
             "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
             "datasets": [
                 {"label": "New Hires", "data": [28, 22, 35, 41, 38, 29, 44, 38, 19]},
                 {"label": "Departures", "data": [18, 14, 22, 19, 24, 17, 21, 28, 11]},
             ]},
            {"type": "doughnut", "title": "Attrition Reasons", "span": 1,
             "labels": ["Better Opportunity", "Compensation", "Culture", "Career Growth", "Other"],
             "datasets": [{"label": "Count", "data": [38, 24, 18, 14, 6]}]},
        ],
        "table": {
            "title": "Department Workforce Summary",
            "headers": ["Department", "Headcount", "Attrition", "Open Roles", "Engagement"],
            "rows": [
                ["Engineering", "520", "9.2%", "28", "76"],
                ["Sales", "380", "18.4%", "22", "68"],
                ["Operations", "440", "11.8%", "14", "71"],
                ["Finance", "180", "8.1%", "8", "79"],
                ["HR", "92", "6.5%", "4", "82"],
                ["Marketing", "230", "14.2%", "8", "74"],
            ],
        },
    }


def _finance_spec(theme: str) -> dict:
    return {
        "title": "Financial Performance Dashboard",
        "subtitle": "Q3 2025 | Consolidated",
        "theme": theme,
        "kpis": [
            {"label": "Total Revenue", "value": "$48.2M", "change": "+11.4%", "trend": "up", "sub": "vs Q3 2024"},
            {"label": "Gross Margin", "value": "64.8%", "change": "+1.2%", "trend": "up", "sub": "vs 63.6%"},
            {"label": "EBITDA", "value": "$14.1M", "change": "+18.2%", "trend": "up", "sub": "29.3% margin"},
            {"label": "Operating Expenses", "value": "$17.4M", "change": "+6.8%", "trend": "down", "sub": "within budget"},
            {"label": "Free Cash Flow", "value": "$9.8M", "change": "+22.5%", "trend": "up", "sub": "strong quarter"},
        ],
        "filters": [
            {"label": "Business Unit", "options": ["All Units", "Product A", "Product B", "Services", "Subscriptions"]},
            {"label": "Period", "options": ["Q3 2025", "Q2 2025", "Q1 2025", "FY 2024"]},
        ],
        "charts": [
            {"type": "bar", "title": "Revenue vs Budget by BU", "span": 1,
             "labels": ["Product A", "Product B", "Services", "Subscriptions"],
             "datasets": [
                 {"label": "Actual ($M)", "data": [18.4, 12.1, 9.8, 7.9]},
                 {"label": "Budget ($M)", "data": [17.0, 13.5, 9.0, 7.5]},
             ]},
            {"type": "line", "title": "Monthly P&L Trend", "span": 2,
             "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
             "datasets": [
                 {"label": "Revenue ($M)", "data": [14.2, 15.1, 16.8, 17.4, 18.1, 17.9, 18.8, 19.2, 10.4]},
                 {"label": "OpEx ($M)", "data": [5.4, 5.6, 5.8, 5.7, 5.9, 5.8, 6.0, 6.1, 3.1]},
             ]},
            {"type": "doughnut", "title": "Revenue Mix", "span": 1,
             "labels": ["Product A", "Product B", "Services", "Subscriptions"],
             "datasets": [{"label": "Revenue", "data": [38, 25, 20, 17]}]},
        ],
        "table": {
            "title": "Business Unit P&L Summary",
            "headers": ["Unit", "Revenue", "vs Budget", "Gross Margin", "EBITDA"],
            "rows": [
                ["Product A", "$18.4M", "+8.2%", "68.4%", "32.1%"],
                ["Product B", "$12.1M", "-10.4%", "58.2%", "24.8%"],
                ["Services", "$9.8M", "+8.9%", "71.2%", "38.4%"],
                ["Subscriptions", "$7.9M", "+5.3%", "84.1%", "51.2%"],
            ],
        },
    }


def _ops_spec(theme: str) -> dict:
    return {
        "title": "Operations Performance Dashboard",
        "subtitle": "Q3 2025 | All Locations",
        "theme": theme,
        "kpis": [
            {"label": "On-Time Delivery", "value": "94.2%", "change": "+1.8%", "trend": "up", "sub": "target: 95%"},
            {"label": "Throughput", "value": "48,210", "change": "+12.4%", "trend": "up", "sub": "units this quarter"},
            {"label": "Defect Rate", "value": "0.84%", "change": "-0.12%", "trend": "up", "sub": "improving"},
            {"label": "Capacity Utilization", "value": "87.3%", "change": "+3.1%", "trend": "up", "sub": "all facilities"},
            {"label": "Cost per Unit", "value": "$12.40", "change": "-$0.80", "trend": "up", "sub": "efficiency gain"},
        ],
        "filters": [
            {"label": "Facility", "options": ["All Facilities", "Plant A", "Plant B", "Plant C", "Warehouse"]},
            {"label": "Shift", "options": ["All Shifts", "Day", "Evening", "Night"]},
        ],
        "charts": [
            {"type": "bar", "title": "Throughput by Facility", "span": 1,
             "labels": ["Plant A", "Plant B", "Plant C", "Warehouse"],
             "datasets": [{"label": "Units", "data": [18400, 14800, 11200, 3810]}]},
            {"type": "line", "title": "Daily Throughput Trend", "span": 2,
             "labels": ["Wk1", "Wk2", "Wk3", "Wk4", "Wk5", "Wk6", "Wk7", "Wk8", "Wk9"],
             "datasets": [{"label": "Units/Day", "data": [1820, 1950, 1840, 2100, 2050, 2180, 2240, 2310, 1920]}]},
            {"type": "doughnut", "title": "Defect Categories", "span": 1,
             "labels": ["Material", "Process", "Equipment", "Human Error"],
             "datasets": [{"label": "Defects", "data": [38, 29, 21, 12]}]},
        ],
        "table": {
            "title": "Facility Performance Summary",
            "headers": ["Facility", "Throughput", "OTD Rate", "Defect Rate", "Utilization"],
            "rows": [
                ["Plant A", "18,400", "96.2%", "0.72%", "91.4%"],
                ["Plant B", "14,800", "93.8%", "0.91%", "86.2%"],
                ["Plant C", "11,200", "92.4%", "0.88%", "82.8%"],
                ["Warehouse", "3,810", "98.1%", "0.31%", "78.4%"],
            ],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/generate")
async def generate_dashboard(req: GenerateRequest):
    """Generate a full dashboard specification from a natural language description."""
    if len(req.description.strip()) < 10:
        raise HTTPException(400, "Description too short. Describe the dashboard you want.")

    if HF_TOKEN:
        spec = generate_spec_hf(req)
    else:
        spec = generate_spec_rule_based(req)

    try:
        validated = DashboardSpec(**spec)
        return validated.dict()
    except Exception as e:
        return spec


@app.post("/generate-from-csv")
async def generate_from_csv(
    description: str,
    theme: str = "#3b82f6",
    file: UploadFile = File(...)
):
    """Generate a dashboard from an uploaded CSV file."""
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    csv_ctx = profile_csv(df)
    req = GenerateRequest(description=description, theme=theme, csv_context=csv_ctx)

    if HF_TOKEN:
        spec = generate_spec_hf(req)
    else:
        spec = generate_spec_rule_based(req)

    return spec


@app.get("/examples")
async def get_examples():
    """Return example prompts for the dashboard generator."""
    return {
        "examples": [
            {"title": "Healthcare Operations", "prompt": "Build a healthcare operations dashboard for a hospital showing patient volume by department, bed occupancy rate, average length of stay, readmission rate, and staff utilization."},
            {"title": "Sales Performance", "prompt": "Create a sales performance dashboard for Q3 2025 showing revenue by region, top products, monthly trend, and team performance metrics."},
            {"title": "HR & Workforce", "prompt": "Design an HR workforce analytics dashboard with headcount by department, attrition rate, time-to-hire, engagement score, and hiring trend."},
            {"title": "Financial Performance", "prompt": "Create a financial P&L dashboard showing revenue vs budget, gross margin, EBITDA, operating expenses, and monthly trend by business unit."},
            {"title": "Operations", "prompt": "Build an operations dashboard showing throughput, on-time delivery rate, defect rate, capacity utilization, and facility performance breakdown."},
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "llm_enabled": bool(HF_TOKEN)}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
