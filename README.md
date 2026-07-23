# ⚡ GreenGrid — Campus Energy Intelligence System
> Data-driven energy anomaly detection and forecasting system for campus and residential environments.

## 🌐 Live Demo
**Try the application here:**
👉 https://greengrid.streamlit.app/

---

## 📖 Overview
GreenGrid addresses a simple problem: KCCITM campus has no sub-meter energy monitoring, so equipment left running after hours or unnoticed power spikes go undetected until the monthly bill arrives. GreenGrid is an AI-driven anomaly-detection and forecasting system that flags abnormal power draw in real time and recommends alerts or occupancy-gated auto-shutoff — validated through data-driven prototyping before any physical sensors are deployed.

---

## 📊 Data Provenance Statement
This repository handles two distinct datasets, used for different purposes. They are never blurred, and every view in the dashboard states which one is active.

| Data Source | Description | Anomaly Labels | Occupancy Sensor |
|---|---|---|---|
| **Synthetic campus data** | 8 rooms, 6 weeks, 5-min intervals, injected anomalies | YES | YES |
| **Real UCI household data** | Single household, 3 circuits, Dec 2006–Nov 2010 | NO | NO |

**Why two datasets:** the synthetic dataset has ground-truth labels, so it's the only place precision/recall/F1 can be legitimately computed. The real UCI dataset has no labels and no occupancy sensor, so it validates the same models against real-world unlabeled noise, but the occupancy-gated auto-shutoff logic can only be tested where occupancy exists — i.e. synthetic data only.

---

## 📈 Validated Results

| Model | Dataset | Metric | Result |
|---|---|---|---|
| Isolation Forest | Synthetic (labeled) | F1 Score | 0.86 |
| Isolation Forest | Real UCI (unlabeled, 409,917 records) | Flagged Rate | ~3.4% (13,808 / 409,917) |
| Prophet | Synthetic | MAE | 0.24 kW |
| Prophet | Real UCI | MAE / RMSE | 0.3375 kW / 0.3799 kW |

Note on the Prophet gap: real household appliance usage has no fixed schedule, unlike a campus timetable — this is expected, and it's actually evidence *for* the design's core assumption that a real campus deployment (following class schedules) should forecast better than this household proxy did.

---

## 🛠️ Tech Stack
- **scikit-learn** — Isolation Forest (anomaly detection)
- **Meta Prophet** — load forecasting
- **Streamlit** — interactive dashboard
- **Plotly** — visualizations (heatmaps, time series, forecast charts)
- **ReportLab** — automated PDF weekly reports

---

## 🏗️ Architecture

Sensing (CT-clamp + PIR, design spec)
→ Edge (Raspberry Pi, design spec)
→ Cloud ingestion (Azure IoT Hub, design spec)
→ ML Engine (Isolation Forest + Prophet, built & validated)
→ Decision Engine (tiered thresholds: >0.85 critical, 0.72–0.85 high-watch, else normal)
→ Action Layer (WhatsApp-style alerts + PDF reports, built)


## 📁 Project Structure

GreenGrid/
├── data/raw/ <- household_power_consumption.txt goes here
├── data/processed/ <- pipeline outputs (CSV/Parquet)
├── models/
│ ├── generate_synthetic_data.py
│ ├── isolation_forest_v2.py
│ ├── prophet_model.py
│ ├── decision_engine.py
│ ├── load_uci_data.py
│ ├── isolation_forest_real_data.py
│ └── prophet_real_data.py
├── dashboard/app.py <- Streamlit UI
├── reports/report_generator.py <- PDF/WhatsApp generation
├── run_pipeline.py <- single-command orchestration
└── requirements.txt


---

## ⚠️ Known Limitations
- **Occupancy-gated auto-shutoff logic is validated on synthetic data only.** The real UCI dataset has no occupancy sensor, so this logic remains a design feature, not one tested against real-world behavior.
- **Impact quantification on the Real UCI view (energy waste, cost saved, CO2e avoided) is under active repair.** The calculation was previously tied to an occupancy flag that doesn't exist in this dataset, causing it to silently read zero despite real anomalies being correctly flagged. Anomaly *detection* on real data is working correctly; impact *quantification* is being fixed to work without a PIR signal.
- The real UCI dataset is a single household, not a multi-room campus — it validates the detection/forecasting *models*, not the multi-room campus structure itself.

---

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
```
No external API keys required. Place `household_power_consumption.txt` in `data/raw/` before running to use real UCI data — if the file is missing, the pipeline will fail loudly rather than silently substituting synthetic data.

---

## 🎓 Project Context
Built as part of the 1M1B Green Skills & Applied AI Internship, KCCITM Greater Noida, 2026.