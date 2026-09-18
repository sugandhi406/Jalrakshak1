# JalRakshak AI — Submit-Ready SDG 6 Prototype

**Domain:** Water Sustainability  
**Primary SDG:** SDG 6 — Clean Water and Sanitation 
**Prototype users:** College/hostel administrators, facility managers, and students  
**Recommended demonstration scope:** **one college/hostel campus**, not a city-wide deployment.

## 1. Problem statement

Water consumption in institutions can become abnormal because of leaking fixtures, tank overflow, pipe seepage, unusual night flow, occupancy changes and unmanaged peak demand. Manual meter checking often identifies the problem only after significant water has been lost.

**JalRakshak AI** converts consumption data into an actionable water-management workflow:

**Consumption data → anomaly detection → leakage-risk alert → forecast → RAG-grounded recommendation → agent-generated maintenance ticket → human inspection**

The system is designed as a decision-support prototype. An anomaly is a **risk signal**, not proof of a physical leak.

## 2. Objectives

1. Improve awareness of water consumption at zone/block level.
2. Identify unusual consumption and persistent night flow early.
3. Forecast near-term demand to support planning.
4. Give practical conservation recommendations grounded in local SOPs.
5. Demonstrate an agentic workflow that converts high-risk alerts into inspection tickets.
6. Accept an optional leakage image for advisory visual triage.
7. Demonstrate a realistic, low-cost architecture that can later connect to smart meters.

## 3. AI/technical components

### Anomaly detection
- Hour-of-day baseline using mean and standard deviation.
- Z-score detection for unusual spikes/dips.
- Night-flow rule for persistent 1–4 AM usage.
- Isolation Forest as a multivariate confirmation signal.
- Severity and confidence are exposed in the alert output.

### Prediction
- Holt-Winters / Exponential Smoothing with weekly seasonality.
- Seven-day demand forecast with an uncertainty band.
- Forecast is calculated independently for each campus zone.

### RAG + Granite/LLM
- Local knowledge base contains SOPs, conservation tips, Parbhani context and previous leak logs.
- TF-IDF retrieval supplies relevant evidence.
- Optional IBM Granite generation is supported through an OpenAI-compatible endpoint (for example, a local Granite deployment).
- If no LLM is configured, the app still works using grounded deterministic responses.

### Agentic workflow
The agent follows:
1. Read detected alerts.
2. Select High-severity events.
3. Check whether an equivalent ticket already exists.
4. Create an inspection ticket.
5. Assign an action window: 4 hours for night-flow signals or 24 hours for spike signals.
6. Facility staff physically verify the issue.
7. Future production version can close the ticket after post-repair meter verification.

### Optional image input
The `/api/leak-image` endpoint accepts an image and returns advisory triage. **The current prototype intentionally labels this as heuristic**; a production pilot should connect a vision-capable Granite/LLM or CV model. The UI explicitly tells the user to confirm with physical inspection and consumption data.

## 4. Prototype data

The included CSVs are simulated but structured like real campus meter data:
- Daily data: 120 days, per zone.
- Hourly data: 30 days, per zone.
- Four zones: boys hostel, girls hostel, day scholars/admin and staff quarters.
- Deliberate anomalies are injected so the demonstration reliably shows the detection pipeline.

This is appropriate for a prototype/demo, but the data must be replaced by actual meter readings for a real deployment.

## 5. Architecture

```text
                 ┌─────────────────────────────┐
                 │ College / Hostel Meter Data │
                 │ CSV / future IoT feed       │
                 └──────────────┬──────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ JalRakshak FastAPI    │
                    └───────────┬───────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
     ┌─────▼─────┐       ┌──────▼──────┐      ┌──────▼──────┐
     │ Anomaly   │       │ Forecasting │      │ RAG         │
     │ Z-score + │       │ Holt-Winters│      │ KB + TF-IDF │
     │ IF + rules│       │ 7-day       │      │ + Granite   │
     └─────┬─────┘       └──────┬──────┘      └──────┬──────┘
           │                    │                    │
           └──────────────┬─────┴──────────────┬─────┘
                          │                    │
                   ┌──────▼──────┐      ┌──────▼─────────┐
                   │ Dashboard   │      │ Agentic Action │
                   │ + Alerts    │      │ Maintenance    │
                   │ + Forecast  │      │ Ticket         │
                   └─────────────┘      └────────────────┘
```

## 6. Run locally

### Backend

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The included data is already present. To regenerate the demo dataset:

```bash
python data_gen.py
```

### Frontend

From the project root:

```bash
cd frontend
python -m http.server 5500
```

Open the local page shown by the server and keep the backend running on port 8000.

## 7. Optional Granite configuration

The backend is ready for an OpenAI-compatible Granite endpoint.

```bash
export GRANITE_API_URL="http://localhost:11434/v1/chat/completions"
export GRANITE_MODEL="granite"
# export GRANITE_API_KEY="..."   # only if your endpoint requires one
uvicorn app:app --reload --port 8000
```

The application will automatically fall back to grounded responses if the model endpoint is unavailable.

## 8. Demo flow for evaluation

Use this 3–5 minute sequence:

1. **Dashboard:** show the Parbhani single-campus scope and four zones.
2. **Alerts:** point out the High-severity night-flow signal in Block A and other anomalies.
3. **Forecast:** switch zones and show the seven-day predicted demand band.
4. **Recommendations:** show the prioritized inspection and conservation actions.
5. **Agentic workflow:** open the generated tickets and explain that detection automatically creates an inspection action.
6. **RAG chat:** ask:
   - “Which block used the most water?”
   - “Is Block A leaking?”
   - “Why has Block C's usage increased?”
   - “What should the facility manager inspect first?”
7. **Image input:** optionally upload a non-sensitive water-fixture/site photo and explain that it is advisory triage.
8. **Impact:** conclude with awareness, earlier abnormal-use detection and data-driven conservation.

## 9. Expected impact

### Direct prototype outcomes
- Better visibility of block-level water use.
- Earlier identification of abnormal consumption.
- Demand planning through short-term forecasts.
- Practical, evidence-grounded conservation actions.
- Faster transition from alert to maintenance inspection.

### Suggested pilot KPIs
For a real college/hostel pilot, measure:
- Litres consumed per resident per day.
- Number of abnormal-use events detected.
- Median time from alert to physical inspection.
- Confirmed leaks / total alerts.
- Estimated water loss avoided after repairs.
- Percentage change in water use after conservation interventions.

These should be measured against the institution's own baseline rather than invented targets.

## 10. Scope and limitations

**In scope:** one college/hostel campus in Parbhani, simulated consumption data, anomaly detection, demand prediction, RAG assistant, optional image triage and agent-generated inspection tickets.

**Out of scope:** city-wide water-network monitoring, automatic valve control, autonomous maintenance, guaranteed leak diagnosis, and production notification infrastructure.

The prototype does **not** claim that an AI alert proves a leak. Physical inspection remains the final verification step.

## 11. Future deployment

1. Connect real smart-meter or manual meter ingestion.
2. Add authentication and role-based views.
3. Store tickets in SQLite/PostgreSQL.
4. Add WhatsApp/email/SMS notifications.
5. Add academic calendar, occupancy and weather features to forecasting.
6. Replace heuristic image triage with a validated vision model.
7. Track post-repair consumption to close the agent loop.
8. Expand the knowledge base with institution-specific SOPs and maintenance records.

## 12. Repository structure

```text
jalrakshak-ai/
├── backend/
│   ├── app.py
│   ├── anomaly.py
│   ├── forecast.py
│   ├── rag.py
│   ├── data_gen.py
│   ├── requirements.txt
│   └── knowledge_base/
├── data/
│   ├── daily_consumption.csv
│   └── hourly_consumption.csv
├── frontend/
│   └── index.html
└── README.md
```

**Submission positioning:** This is a focused proof-of-concept for SDG 6. The strength of the project is the complete path from consumption data to risk detection, prediction, grounded advice and human-in-the-loop action — without requiring city-scale infrastructure.
