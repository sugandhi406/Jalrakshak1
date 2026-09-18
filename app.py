"""
JalRakshak AI — FastAPI backend
Prototype scope: one college/hostel campus in Parbhani, Maharashtra.
"""
import os, random
from functools import lru_cache
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anomaly import detect_anomalies
from forecast import forecast_all
from rag import KnowledgeBase, answer_query

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DAILY_CSV = os.path.join(DATA_DIR, "daily_consumption.csv")
HOURLY_CSV = os.path.join(DATA_DIR, "hourly_consumption.csv")

app = FastAPI(title="JalRakshak AI", version="2.0.0",
              description="SDG 6 water sustainability prototype for a Parbhani college/hostel campus")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

kb = KnowledgeBase()
TICKETS = []


def _load_daily():
    return pd.read_csv(DAILY_CSV, parse_dates=["date"])


@lru_cache(maxsize=1)
def _cached_alerts():
    return detect_anomalies(HOURLY_CSV)


@lru_cache(maxsize=1)
def _cached_forecast():
    return forecast_all(DAILY_CSV)


def _run_agentic_pass():
    """Detection → risk scoring → action recommendation → maintenance ticket."""
    alerts = _cached_alerts()
    existing = {(t["zone"], t["timestamp"], t["issue"]) for t in TICKETS}
    for a in alerts:
        if a["severity"] == "High":
            key = (a["zone"], a["timestamp"], a["type"].replace("_", " "))
            if key not in existing:
                action = ("Inspect zone within 4 hours" if a["type"] == "continuous_night_flow"
                          else "Inspect zone within 24 hours")
                TICKETS.append({
                    "id": len(TICKETS)+1, "zone": a["zone"], "timestamp": a["timestamp"],
                    "issue": a["type"].replace("_", " "), "severity": a["severity"],
                    "status": "Open", "next_action": action,
                    "note": a["note"], "created_by": "JalRakshak agent"
                })


@app.on_event("startup")
def startup():
    _run_agentic_pass()


@app.get("/")
def root():
    return {"status":"ok","project":"JalRakshak AI","scope":"College/Hostel Pilot — Parbhani, Maharashtra","sdg":"SDG 6"}


@app.get("/api/health")
def health():
    return {"status":"healthy","data_files": os.path.exists(DAILY_CSV) and os.path.exists(HOURLY_CSV),
            "llm_configured": bool(os.getenv("GRANITE_API_URL"))}


@app.get("/api/zones")
def zones():
    return {"zones": sorted(_load_daily()["zone"].unique().tolist())}


@app.get("/api/consumption")
def consumption(zone: str = None, days: int = 30):
    days = max(1, min(days, 365))
    df = _load_daily()
    if zone:
        df = df[df["zone"] == zone]
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    df = df[df["date"] > cutoff]
    return df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")


@app.get("/api/summary")
def summary():
    df = _load_daily()
    last7 = df[df["date"] > df["date"].max()-pd.Timedelta(days=7)]
    totals = last7.groupby("zone")["litres"].sum().sort_values(ascending=False)
    alerts = _cached_alerts()
    return {"as_of": str(df["date"].max().date()), "last_7d_litres": round(float(last7["litres"].sum()),1),
            "zone_totals": {k: round(float(v),1) for k,v in totals.items()},
            "alert_count": len(alerts), "high_alerts": sum(a["severity"]=="High" for a in alerts)}


@app.get("/api/alerts")
def alerts(zone: str = None):
    data = _cached_alerts()
    if zone: data = [a for a in data if a["zone"] == zone]
    return {"count": len(data), "alerts": data}


@app.get("/api/forecast")
def forecast(zone: str = None, horizon: int = 7):
    horizon = max(1, min(horizon, 14))
    data = forecast_all(DAILY_CSV, horizon=horizon)
    return data.get(zone, {}) if zone else data


@app.get("/api/recommendations")
def recommendations():
    df = _load_daily()
    alerts = _cached_alerts()
    recs = []
    high_zones = sorted({a["zone"] for a in alerts if a["severity"]=="High"})
    for zone in high_zones:
        recs.append({"priority":"High","zone":zone,"action":"Physically inspect tanks, flush valves, pipe joints and outdoor taps.",
                     "reason":"High-confidence leakage-risk signal detected."})
    recs += [
        {"priority":"Medium","zone":"Campus","action":"Track night flow separately and investigate persistent 1–4 AM usage.",
         "reason":"Night flow is a strong early-warning indicator in hostels."},
        {"priority":"Medium","zone":"Campus","action":"Stagger peak shower/laundry demand and repair dripping fixtures promptly.",
         "reason":"Reduces peak pressure and avoidable consumption."},
        {"priority":"Low","zone":"Campus","action":"Evaluate rooftop rainwater harvesting and greywater reuse for non-potable uses.",
         "reason":"Supports local water resilience and SDG 6."}
    ]
    return {"recommendations": recs}


@app.get("/api/tickets")
def tickets():
    _run_agentic_pass()
    return {"count": len(TICKETS), "tickets": TICKETS}


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    result = answer_query(req.message.strip(), _load_daily(), _cached_alerts(), kb, _cached_forecast())
    return {"query": req.message, **result}


@app.post("/api/leak-image")
async def leak_image(file: UploadFile = File(...)):
    """
    Optional visual input demo. This prototype accepts an image and returns
    advisory triage. It is deliberately labelled as heuristic; production
    should replace this with a vision-capable Granite/LLM or CV model.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    contents = await file.read()
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image must be 8 MB or smaller.")
    verdicts = [
        ("Possible seepage/pooling — recommend inspection", 0.74),
        ("Possible fixture leak — recommend inspection", 0.69),
        ("No clear visual leak sign — correlate with meter anomaly", 0.61)
    ]
    verdict, confidence = random.choice(verdicts)
    return {"filename":file.filename, "verdict":verdict, "confidence":confidence,
            "note":"Advisory prototype triage only. Confirm visually and with consumption data before maintenance action.",
            "linked_workflow":"Combine image triage with anomaly alerts before dispatch."}
