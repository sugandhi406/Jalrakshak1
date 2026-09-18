"""
JalRakshak AI — RAG + Granite/LLM assistant.

The assistant is intentionally provider-agnostic:
- RAG retrieval: local TF-IDF over campus SOPs, conservation guidance and leak history.
- Live tools: consumption summary, anomaly alerts and 7-day demand forecast.
- LLM: optional IBM Granite via an OpenAI-compatible endpoint (default: local Ollama).
- Safe fallback: deterministic grounded response when no LLM is configured.

No model is allowed to invent measurements; numeric claims come from live tools.
"""
import os, re, glob, json
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


class KnowledgeBase:
    def __init__(self, kb_dir=KB_DIR):
        self.chunks, self.sources = [], []
        for path in glob.glob(os.path.join(kb_dir, "*.txt")):
            text = open(path, encoding="utf-8").read()
            for block in re.split(r"\n\s*\n", text):
                block = block.strip()
                if block:
                    self.chunks.append(block)
                    self.sources.append(os.path.basename(path))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.chunks) if self.chunks else None

    def retrieve(self, query, k=4):
        if not self.chunks or self.matrix is None:
            return []
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix).flatten()
        top = sims.argsort()[::-1][:k]
        return [{"text": self.chunks[i], "source": self.sources[i], "score": round(float(sims[i]), 3)}
                for i in top if sims[i] > 0.03]


def find_zone_mention(query, zones):
    q = query.lower()
    for zone in zones:
        short = zone.split(" (")[0].lower()
        if short in q or zone.lower() in q:
            return zone
    return None


def _live_tools(query, daily_df, alerts, forecast_data):
    zones = sorted(daily_df["zone"].unique().tolist())
    zone = find_zone_mention(query, zones)
    q = query.lower()
    latest = daily_df["date"].max()
    last7 = daily_df[daily_df["date"] > latest - pd.Timedelta(days=7)]
    facts = []

    if any(x in q for x in ["most water", "highest usage", "used the most", "top consumer"]):
        totals = last7.groupby("zone")["litres"].sum().sort_values(ascending=False)
        facts.append(f"Last 7-day totals: " + ", ".join(f"{z}: {v:,.0f} L" for z, v in totals.items()))
        facts.append(f"Highest 7-day consumer: {totals.index[0]} at {totals.iloc[0]:,.0f} L.")

    if any(x in q for x in ["leak", "leaking", "anomaly", "risk", "alert"]):
        rel = [a for a in alerts if zone is None or a["zone"] == zone]
        if rel:
            for a in rel[:3]:
                facts.append(f"Alert: {a['zone']} | {a['severity']} | {a['type']} | {a['timestamp']} | {a['note']}")
        else:
            facts.append(f"No active leakage-risk alerts detected for {zone or 'the campus'}.")

    if zone and any(x in q for x in ["why", "increase", "higher", "trend", "compare", "usage"]):
        zdf = daily_df[daily_df["zone"] == zone].sort_values("date")
        recent = zdf.tail(7)["litres"].mean()
        prior = zdf.iloc[-14:-7]["litres"].mean() if len(zdf) >= 14 else recent
        pct = ((recent-prior)/prior*100) if prior else 0
        facts.append(f"{zone}: average daily use is {recent:,.0f} L/day vs {prior:,.0f} L/day in the preceding week ({pct:+.1f}%).")

    if any(x in q for x in ["forecast", "predict", "tomorrow", "next week", "demand"]):
        z = zone or zones[0]
        fc = forecast_data.get(z, {}).get("forecast", [])[:7]
        if fc:
            facts.append(f"7-day forecast for {z}: " + ", ".join(f"{x['date']} {x['litres']:,.0f} L" for x in fc))

    if not facts:
        totals = last7.groupby("zone")["litres"].sum()
        facts.append("Campus 7-day snapshot: " + ", ".join(f"{z}: {v:,.0f} L" for z,v in totals.items()))
        facts.append(f"Current anomaly/leakage-risk alerts: {len(alerts)}.")

    return facts, zone


def answer_query(query, daily_df, alerts, kb, forecast_data=None):
    forecast_data = forecast_data or {}
    facts, zone = _live_tools(query, daily_df, alerts, forecast_data)
    retrieved = kb.retrieve(query + " " + (zone or ""), k=4)
    answer, provider = _compose_with_granite(query, facts, retrieved)
    if answer:
        return {"answer": answer, "provider": provider, "sources": [r["source"] for r in retrieved]}
    return {"answer": _compose_template(facts, retrieved), "provider": "grounded-fallback", "sources": [r["source"] for r in retrieved]}


def _compose_with_granite(query, facts, retrieved):
    """Call an OpenAI-compatible Granite endpoint. Works with local Ollama by default."""
    endpoint = os.getenv("GRANITE_API_URL", "http://localhost:11434/v1/chat/completions")
    model = os.getenv("GRANITE_MODEL", "granite")
    api_key = os.getenv("GRANITE_API_KEY", "")
    system = """You are JalRakshak AI, a water-sustainability assistant for a college/hostel pilot in Parbhani, Maharashtra.
Answer only from the LIVE FACTS and REFERENCE CONTEXT. Never invent measurements, dates, leak causes, or savings.
Explain uncertainty: an anomaly is a leakage-risk signal, not proof of a leak. Give one practical next action.
Keep answers concise (3-6 sentences)."""
    prompt = {
        "model": model, "temperature": 0.1, "max_tokens": 350,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":f"QUESTION: {query}\n\nLIVE FACTS:\n" + "\n".join("- "+x for x in facts) +
             "\n\nREFERENCE CONTEXT:\n" + "\n---\n".join(r["text"] for r in retrieved)}
        ]
    }
    try:
        headers = {"Content-Type":"application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.post(endpoint, headers=headers, json=prompt, timeout=12)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        if text:
            return text, f"Granite ({model})"
    except Exception:
        pass
    return None, None


def _compose_template(facts, retrieved):
    text = " ".join(facts)
    if retrieved:
        first = retrieved[0]["text"].split("\n")[0]
        text += f" Related guidance: {first}"
    return text
