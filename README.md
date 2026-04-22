# S2.T1.2 · Stunting Risk Heatmap Dashboard

**AIMS KTT Hackathon · Tier 1 · HealthTech / Geospatial / Data Viz**

> Childhood stunting affects ~33% of Rwandan children under 5 (NISR DHS 2019–20).
> This dashboard helps Abunzi and Umudugudu community leaders identify the highest-risk
> households in their sector — and puts a printed action page in their hands monthly.

---

## Quick start (≤ 2 commands, free Colab CPU)

```bash
pip install -r requirements.txt
python generate_data.py && python risk_scorer.py && python export_printable.py && streamlit run dashboard.py
```

> All scripts run CPU-only. Dashboard renders in < 3 s on Colab.

---

## Repository layout

```
├── generate_data.py       # Reproducible synthetic data generator (seed 42)
├── risk_scorer.py         # Hybrid logistic-regression + rule-based scorer
├── dashboard.py           # Streamlit interactive dashboard
├── export_printable.py    # A4 PDF generator (one per sector, bilingual)
├── data/                  # Created by generate_data.py
│   ├── households.csv     # 2,500 households (10 features)
│   ├── gold_stunting_flag.csv  # 300 labelled rows (150/150)
│   ├── districts.geojson  # 5 district polygons
│   └── sectors.geojson    # 15 sector polygons
├── output/                # Created by risk_scorer.py
│   ├── households_scored.csv
│   ├── sector_summary.csv
│   ├── scorer.pkl
│   └── metrics.json
└── printable/             # Created by export_printable.py (15 PDFs)
```

---

## Model performance

| Metric    | Value  |
|-----------|--------|
| AUC-ROC   | 0.935  |
| Precision | 0.8387 |
| Recall    | 0.8667 |
| F1        | 0.8525 |
| N train   | 240    |
| N test    | 60     |

> Metrics evaluated on a **held-out 20% test set** (stratified split, never seen
> during training). Logistic regression with Platt scaling on 5 features: water
> source, sanitation tier, income band, meal frequency, children under 5.
> Features match the DHS stunting predictors identified in Rwanda DHS 2019–20
> (NISR/ICF 2021).

**Biggest driver: water source quality** — the single feature with the highest
model weight (0.30). Households on unprotected wells or river water carry
~2.5× the risk of piped-water households, controlling for other features.

### Why Nyarugenge < Gasabo (choropleth gap)

Nyarugenge is Kigali's densest urban core (90% urban households in the
synthetic data, reflecting real NISR urbanisation figures).  
Gasabo is 80% urban but contains peri-urban sectors (Kimironko/Remera) where
unprotected-well use climbs to ~25% vs. Nyarugenge's ~5%. That water-source
gap (~20 pp) drives most of the risk difference. Income band contributes
secondarily (more "very low" households in Gasabo peri-urban areas).

---

## Real NISR stunting baselines (Rwanda DHS 2019–20)

| Province / City  | Stunting prevalence |
|------------------|---------------------|
| Kigali City      | ~18–22%             |
| Northern Province| ~40%                |
| Southern Province| ~38%                |
| Eastern Province | ~35%                |
| Western Province | ~30%                |

> Source: NISR / ICF. *Rwanda Demographic and Health Survey 2019–20.*
> Kigali 2021. Table 11.1 (stunting in children under 5).

The synthetic data models Nyarugenge/Gasabo/Kicukiro on Kigali baselines,
Musanze on Northern Province, and Nyanza on Southern Province.

---

## Product & Business Adaptation

### The paper-first workflow (for village chiefs without laptops)

**Who gets it:** One A4 sheet per Umudugudu sector, printed at the sector
health post and handed to the Umudugudu chief at the monthly community meeting.

**What is on the sheet (top to bottom):**

1. **Header:** MINISANTE logo · Sector name · Month/Year · "IBANGA" (CONFIDENTIAL) warning
2. **Summary box:** Total households | High-risk count | Avg risk score | % high-risk
3. **Top-10 table (anonymised):** Households identified only by a 6-character
   code (e.g. `GIT-03`), never by name. Columns: rank, code, score (0–1),
   children under 5, top-3 risk drivers, one-line intervention hint.
4. **Annotation block (blank lines):** Chief signs, notes follow-up actions,
   checks the MINISANTE escalation box.
5. **Legend:** Red/Orange/Yellow/Green risk bands explained in Kinyarwanda.

**Privacy:** Household ID codes are salted with a monthly nonce stored only
at district health office. The chief cannot reverse-map a code to a household
without the sector health worker's key — protecting family privacy while
enabling follow-up.

### Monthly district meeting loop

| Step | Who | Action |
|------|-----|--------|
| Day 1 | District data officer | Runs `python risk_scorer.py && python export_printable.py`, prints 15 sheets |
| Day 2–3 | Sector health workers | Distribute sheets to Umudugudu chiefs at sector meetings |
| Day 4–25 | Umudugudu chief | Visits households on the list, annotates the sheet |
| Day 26 | Sector health worker | Collects annotated sheets, notes any household with score ≥ 0.75 |
| Day 28 | Sector health worker | Escalates critical cases (score ≥ 0.75, 3+ children) to MINISANTE via SMS |
| Day 30 | District health officer | Aggregates annotations, updates household visit log |

**SMS escalation (no smartphone needed):**  
Africa's Talking IVR/SMS gateway. Daily digest to the sector health worker:
> "Sector Gitega: 3 households critical this month. Codes GIT-01, GIT-04,
> GIT-07. Reply YES to confirm escalation."

**Low-bandwidth / offline:**  
The Streamlit dashboard can be run once monthly over 3G to refresh data, then
the PDFs are the offline artefact. The scorer + PDF generator runs fully
offline (no internet required after `pip install`).

### Intervention hints (stretch goal — implemented)

Each high-risk household row includes a one-line action:
- **Water source** → "WASH upgrade — connect to protected/piped water source"
- **Sanitation** → "Sanitation — install improved latrine (VIP/pour-flush)"
- **Income** → "Social protection — refer to Ubudehe/cash-transfer programme"
- **Meal frequency** → "Nutrition — enroll in supplementary feeding (RUTF/Imbuto)"
- **Children under 5** → "Referral — CHW multi-child nutrition screening"

---

## Key design decisions

**1. Threshold set at 0.50, not optimised for recall.**
A higher-recall threshold would flag more households as high-risk, but in a monthly paper workflow where a village chief visits households personally, over-flagging erodes community trust faster than missing a borderline case. 0.50 (the LR probability midpoint) keeps the list actionable and credible.

**2. Privacy-by-design anonymisation.**
Printed sheets identify households only by a 6-character code (e.g. `GIT-03`) generated with a monthly salt stored exclusively at the district health office. The village chief who receives the sheet cannot reverse-map any code to a family name — protecting household privacy while still enabling targeted follow-up by the sector health worker who holds the key.

**3. Offline-first, paper as the primary artefact.**
The dashboard requires internet only once per month to refresh scores. After that, the 15 printed A4 sheets are the sole working artefact — no laptop, no connectivity, no smartphone required for community health workers or village chiefs. This matches the real operational context of Rwanda's Umudugudu system.

---

## 4-minute video

> [Watch 4-minute walkthrough video](https://www.tella.tv/video/stunting-risk-video-1-7zi4)

Segments: on-camera intro (0:00–0:30) · risk_scorer.py walkthrough (0:30–1:30)
· dashboard live demo (1:30–2:30) · A4 PDF walkthrough (2:30–3:30)
· 3 questions answered aloud (3:30–4:00).

---

## Process log

See [process_log.md](process_log.md).

## Honor code

See [SIGNED.md](SIGNED.md).
