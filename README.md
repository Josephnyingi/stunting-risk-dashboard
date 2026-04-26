# S2.T1.2 · Stunting Risk Heatmap Dashboard

**AIMS KTT Hackathon · Tier 1 · HealthTech / Geospatial / Data Viz**
**Author: Joseph Nyingi Wambua**

---

## 🚀 Live Demo & Model

| | Link |
|---|---|
| **▶ Interactive Dashboard** | **https://huggingface.co/spaces/Nyingi101/stunting-risk-heatmap** |
| **Model (scorer.pkl + card)** | https://huggingface.co/Nyingi101/stunting-risk-scorer |
| **4-minute video** | https://youtu.be/znyDiG5o8ow |

![Rwanda Childhood Stunting Risk Heatmap Dashboard](https://huggingface.co/Nyingi101/stunting-risk-scorer/resolve/main/dashboard_screenshot.jpg)

---

## Problem Statement

Childhood stunting — defined as height-for-age z-score below −2 SD — affects approximately **33% of Rwandan children under 5** (NISR DHS 2019–20). Stunting is largely irreversible after age 2 and causes life-long cognitive and physical impairment.

Community leaders — **Abunzi** (local arbitrators) and **Umudugudu chiefs** (village administrators) — are the frontline of Rwanda's community health system. They know their households by name but have no data tool to tell them *which families to prioritise* each month. Most have no laptop, no smartphone, and intermittent power.

**This project builds two things:**
1. A web dashboard (Streamlit + Plotly choropleth) for district health officers to explore sector-level risk.
2. A paper-first delivery channel: a bilingual A4 printout per sector, designed for a village chief who reads it at a monthly meeting and annotates it by hand.

---

## Approach — Three-Phase Pipeline

```
Phase 1: DATA             Phase 2: SCORE            Phase 3: DELIVER
─────────────────         ──────────────────         ────────────────────
generate_data.py    →     risk_scorer.py      →      dashboard.py
  2,500 households          Logistic regression        Choropleth map
  300 gold labels           (Platt-calibrated,         Threshold slider
  5 district GeoJSONs       80/20 test split)          District filter
  15 sector GeoJSONs        Per-household score        
                            Sector aggregation  →      export_printable.py
                            metrics.json               15 bilingual A4 PDFs
                                                       Top-10 + interventions
```

### Phase 1 · Data Generation (`generate_data.py`)

The generator creates synthetic NISR-style data seeded at 42 (fully reproducible). Each household receives 10 features sampled conditional on district and urban/rural flag:

| Feature | Values | Risk encoding |
|---------|--------|---------------|
| `water_source` | piped / protected_well / unprotected_well / river_lake | 0.0 → 1.0 |
| `sanitation_tier` | improved / basic / limited / open_defecation | 0.0 → 1.0 |
| `income_band` | high / medium / low / very_low | 0.0 → 1.0 |
| `avg_meal_count` | 1–5 meals/day | inverted: 5 meals → 0.0 risk |
| `children_under5` | 0–6 | capped at 5, normalised → 0.0–1.0 |

District stunting baselines match NISR DHS 2019–20 (Kigali ~18–22%, Northern Province ~40%, Southern Province ~38%).  
Urban/rural split: Nyarugenge 92.5% urban, Musanze 26.7% urban, Nyanza 13.8% urban — matching real NISR urbanisation figures.

**Gold labels (300 households, 150 positive / 150 negative):** sampled from overlapping probability ranges (positive ≥ 0.32, negative ≤ 0.42) to reflect real survey ambiguity — borderline households appear in both classes, making classification non-trivial.

### Phase 2 · Risk Scoring (`risk_scorer.py`)

**Model choice:** Logistic regression with Platt scaling (CalibratedClassifierCV, 5-fold).  
*Why not a tree or neural network?* This is a Tier 1 challenge. LR gives interpretable, auditable coefficients — essential for community health credibility. A village chief or district officer must be able to challenge a score ("why is household X flagged?"). LR weights provide that answer in one sentence. Platt scaling ensures probabilities are well-calibrated so the 0.50 threshold is meaningful.

**Feature engineering (`featurize`):** Each row is mapped to a 5-dimensional `[0,1]` vector. This is the entire feature transform — no one-hot encoding, no embeddings. The simplicity is intentional: fewer features = fewer ways to overfit on 300 gold labels.

**Training / evaluation:** 80/20 stratified split (240 train, 60 test). Metrics reported on the **held-out 20%** — never seen during training.

**Threshold calibration:** Default 0.50 (LR probability midpoint). *Why not tune for recall?* In a paper-based monthly workflow where a village chief physically visits each flagged household, over-flagging erodes community trust faster than missing a borderline case. 0.50 is the honest midpoint; the dashboard slider lets a district officer tighten or relax it.

### Phase 3 · Delivery (`dashboard.py` + `export_printable.py`)

- **Dashboard:** Plotly choropleth with `carto-positron` tiles (no Mapbox token), Streamlit `@st.cache_data` for < 3 s renders. Tabs: choropleth map, sector table, household table, risk driver charts.
- **PDFs:** ReportLab A4, no external fonts, bilingual English/Kinyarwanda. 15 PDFs (one per sector). Households anonymised as `GIT-01` codes. Annotation and escalation blocks printed on every sheet.

---

## Quick Start — ≤ 2 commands (free Colab CPU)

```bash
pip install -r requirements.txt
python generate_data.py && python risk_scorer.py && python export_printable.py
streamlit run dashboard.py
```

To reproduce the full analysis and validate all figures:

```bash
python analysis.py
```

> All scripts run CPU-only. `streamlit run dashboard.py` renders in < 3 s on Colab.

---

## Repository Layout

```
├── generate_data.py       # Phase 1: reproducible synthetic data (seed 42)
├── risk_scorer.py         # Phase 2: LR scorer, trains + scores all 2,500 HH
├── dashboard.py           # Phase 3: Streamlit choropleth dashboard
├── export_printable.py    # Phase 3: A4 PDF generator (bilingual, anonymised)
├── analysis.py            # Standalone results validation — run to reproduce all figures
├── requirements.txt
├── data/                  # Created by generate_data.py
│   ├── households.csv         # 2,500 rows × 10 features
│   ├── gold_stunting_flag.csv # 300 labelled rows (150 positive / 150 negative)
│   ├── districts.geojson      # 5 district polygons
│   └── sectors.geojson        # 15 sector polygons
├── output/                # Created by risk_scorer.py
│   ├── households_scored.csv  # + risk_score, risk_tier, top_drivers, intervention
│   ├── sector_summary.csv     # 15 rows: avg_score, pct_high_risk, n_households
│   ├── scorer.pkl             # Serialised model (CalibratedClassifierCV + scaler)
│   └── metrics.json           # Test-set metrics (AUC-ROC, P, R, F1)
└── printable/             # Created by export_printable.py — 15 A4 PDFs
    └── sector_<name>.pdf
```

---

## Results

### Model Performance

| Metric | Value | What it means |
|--------|-------|---------------|
| **AUC-ROC** | **0.935** | Model correctly ranks a high-risk HH above a low-risk HH 93.5% of the time on unseen data |
| Precision | 0.839 | 84% of households flagged as high-risk genuinely have high-risk characteristics |
| Recall | 0.867 | The model catches 87% of all high-risk households in the test set |
| F1 | 0.853 | Balanced precision/recall performance |
| Train N | 240 | Gold-label households used for training (80% stratified) |
| Test N | 60 | Held-out households used for evaluation — never seen during training |

### Feature Importance

The logistic regression identifies these coefficients on the standardised feature matrix:

| Rank | Feature | LR Coefficient | Rule Weight | Practical meaning |
|------|---------|---------------|------------|-------------------|
| 1 | Sanitation access | +1.666 | 0.25 | Open defecation → most discriminative in LR |
| 2 | Water source quality | +1.536 | 0.30 | River/lake water → 2nd most discriminative |
| 3 | Income level | +1.070 | 0.25 | Very-low income compounds all other risks |
| 4 | Meal frequency | +0.801 | 0.12 | Fewer than 2 meals/day adds meaningful risk |
| 5 | Children under 5 | +0.516 | 0.08 | More children = more exposure, less per-child care |

**Biggest single driver: sanitation access** (LR coefficient 1.666). Households with open defecation are flagged critical at far higher rates than any other group. Water source is the largest *rule-based* weight (0.30) because literature assigns it higher prior importance; sanitation edges ahead in the LR because the feature separation is sharper in this synthetic dataset.

### District-Level Risk Breakdown

| District | N | Avg score | % High-risk | % Critical | % Urban |
|----------|---|-----------|-------------|------------|---------|
| **Nyarugenge** | 600 | 0.338 | **29.8%** | 18.2% | 92.5% |
| **Gasabo** | 650 | 0.401 | **35.4%** | 25.4% | 79.4% |
| **Kicukiro** | 550 | 0.420 | **38.4%** | 28.0% | 76.2% |
| **Musanze** | 300 | 0.728 | **74.0%** | 66.7% | 26.7% |
| **Nyanza** | 400 | 0.821 | **87.3%** | 76.3% | 13.8% |

### Sector Ranking (All 15 Sectors)

| Rank | District | Sector | % High-risk | Avg score |
|------|----------|--------|-------------|-----------|
| 1 | Nyanza | **Kibilizi** | **93.0%** | 0.855 |
| 2 | Nyanza | **Busasamana** | **85.0%** | 0.801 |
| 3 | Nyanza | **Cyabakamyi** | **82.9%** | 0.801 |
| 4 | Musanze | **Muhoza** | **79.4%** | 0.766 |
| 5 | Musanze | **Shingiro** | **73.5%** | 0.737 |
| 6 | Musanze | Kinigi | 68.4% | 0.675 |
| 7 | Kicukiro | Gahanga | 43.5% | 0.448 |
| 8 | Kicukiro | Niboye | 40.7% | 0.451 |
| 9 | Gasabo | Kacyiru | 37.2% | 0.418 |
| 10 | Gasabo | Kimironko | 34.6% | 0.384 |
| 11 | Gasabo | Remera | 34.3% | 0.403 |
| 12 | Nyarugenge | Gitega | 32.0% | 0.350 |
| 13 | Kicukiro | Masaka | 30.3% | 0.358 |
| 14 | Nyarugenge | Nyarugenge | 29.8% | 0.356 |
| 15 | **Nyarugenge** | **Kigali** | **27.8%** | **0.309** |

---

## Interpreting the Results

### Why does Nyarugenge differ from Gasabo? (choropleth gap)

Nyarugenge is Kigali's densest urban core — **92.5% urban** in the synthetic data (matching NISR figures). Urban households in this dataset have:
- 55% piped-water access vs. 10% in rural areas
- 55% improved sanitation vs. 10% in rural areas
- Higher income band concentration

Gasabo is **79.4% urban** but contains peri-urban sectors (Kimironko, Remera) where the water source mix shifts: ~25% unprotected-well use vs. Nyarugenge's ~5%. That water-source gap drives most of the 5.6 percentage-point difference in high-risk prevalence (29.8% → 35.4%). Income band contributes secondarily.

**The single feature that explains most of the gap:** unprotected well usage — a household on an unprotected well is 59.2% likely to be flagged critical, vs 8.0% for a piped-water household (verified by `analysis.py` section 5).

### The stark rural/urban divide in water source

| Water source | % Critical | % Low risk |
|-------------|-----------|------------|
| **River / lake** | **87.6%** | 2.8% |
| Unprotected well | 59.2% | 17.8% |
| Protected well | 33.9% | 44.0% |
| **Piped** | **8.0%** | **78.7%** |

River-lake users are 10.95× more likely to be critical than piped-water users. This is the dominant driver of the Nyanza/Musanze rural cluster's high prevalence.

### Are the results coherent with NISR baselines?

**Important distinction:** the model outputs a *risk flag* (score ≥ 0.50 = household has multiple risk factors). This is NOT equivalent to measured stunting prevalence (height-for-age z-score ≤ −2 SD in children). The model is a *screening tool*, not a prevalence estimator.

| District | % Flagged (model) | NISR stunting prevalence | Ordering preserved? |
|----------|-------------------|--------------------------|---------------------|
| Nyarugenge | 29.8% | ~20% (Kigali) | ✅ Lowest |
| Gasabo | 35.4% | ~21% (Kigali) | ✅ Low |
| Kicukiro | 38.4% | ~24% (Kigali) | ✅ Low |
| Musanze | 74.0% | ~41% (Northern) | ✅ High |
| Nyanza | 87.3% | ~38% (Southern) | ✅ Highest |

The model correctly preserves the **ordinal ranking** from NISR: rural > peri-urban > urban. The flagging rates are higher than NISR prevalence because the model flags *risk factors*, not confirmed stunting — a household can have river water and open defecation without a stunted child (the flagging rate is a risk signal, not a case count).

---

## Product & Business Adaptation

### The Paper-First Workflow

A village chief without a laptop receives a printed A4 sheet each month. The exact layout:

| Position | Content |
|----------|---------|
| Top | "MINISANTE / NISR · Monthly Report" · Sector · Date · **IBANGA (CONFIDENTIAL)** warning in red |
| Summary box | Total HH · High-risk count · Avg score · % high-risk (bilingual) |
| Main table | Top 10 households, ranked by risk score. Columns: rank · anonymised code · score (colour-coded) · children <5 · top-3 risk drivers · intervention hint |
| Bottom | Signature lines · annotation field · MINISANTE escalation checkbox |
| Footer | Colour legend in Kinyarwanda |

**Anonymisation:** Each household is printed as a code (e.g. `KIL-03`) — sector prefix + rank. The mapping from code to household ID is held only at the sector health post, protected by a monthly nonce. The chief cannot reverse-map without the health worker's key.

### Monthly District Meeting Loop

| Day | Actor | Action |
|-----|-------|--------|
| 1 | District data officer | `python risk_scorer.py && python export_printable.py` — prints 15 sheets |
| 2–3 | Sector health workers | Hand sheets to Umudugudu chiefs at sector meetings |
| 4–25 | Umudugudu chief | Visits flagged households; annotates sheet with observations |
| 26 | Sector health worker | Collects sheets; escalates score ≥ 0.75 households |
| 28 | Sector health worker | SMS via Africa's Talking: *"Sector Gitega: 3 critical HH. Codes GIT-01, GIT-04, GIT-07. Reply YES to confirm escalation."* |
| 30 | District health officer | Aggregates annotations; updates household visit log |

**Offline / low-bandwidth:** The Streamlit dashboard needs internet once per month to refresh scores (3G minimum). PDFs are the offline artefact — the scorer and PDF generator run fully offline after `pip install`.

### Intervention Hints per High-Risk Household *(stretch goal — implemented)*

| Primary driver | One-line hint printed on A4 |
|----------------|----------------------------|
| Water source | WASH upgrade — connect to protected/piped water source |
| Sanitation | Install improved latrine (VIP/pour-flush) |
| Income | Refer to Ubudehe / cash-transfer programme |
| Meal frequency | Enroll in supplementary feeding (RUTF / Imbuto programme) |
| Children <5 | CHW multi-child nutrition screening |

---

## Key Design Decisions and Trade-offs

**1. Logistic regression, not a tree ensemble.**  
A gradient-boosted model would score higher on AUC with this data. But: (a) 300 gold labels is too few for reliable tree ensembles; (b) LR coefficients are directly auditable in a live defense or community meeting; (c) render time on Colab stays < 3 s without heavy sklearn overhead.

**2. Threshold at 0.50, not recall-optimised.**  
A threshold of 0.35 would recall ~98% of high-risk households but flag ~60% of all households — making the top-10 A4 page meaningless (every household would qualify). The 0.50 midpoint keeps the flagging rate actionable: 29–87% per district, reflective of real risk-factor load.

**3. Plotly `choropleth_map` instead of Folium.**  
Folium requires `streamlit-folium` (an extra dependency) and serialises HTML into the Streamlit iframe. Plotly renders the choropleth natively in Streamlit with no extra package, uses vector rendering, and handles the GeoJSON `featureidkey` match in one parameter.

**4. Sector-level polygons generated from bounding boxes.**  
The brief provides district-level GeoJSON. To show sector-level variation, each district is subdivided into 3 non-overlapping rectangular sectors in `sectors.geojson`. This is declared as synthetic geometry; in production it would be replaced with NISR official sector boundaries.

**5. Privacy-by-design anonymisation.**  
Household IDs on paper are sector-prefix codes (`KIL-03`), not the internal `HH00123` IDs. The mapping is held only at the district health office. This follows Rwanda's 2021 Data Protection Law, which requires anonymisation for health micro-data shared beyond the district level.

---

## Real NISR Stunting Baselines (Rwanda DHS 2019–20)

| Province / City | Stunting prevalence | Notes |
|----------------|---------------------|-------|
| Kigali City | ~18–22% | Lowest in country; high urbanisation, better water/sanitation |
| Northern Province | ~40% | Musanze, Rulindo — high altitude, subsistence agriculture |
| Southern Province | ~38% | Nyanza, Huye — predominantly rural |
| Eastern Province | ~35% | Rwamagana, Kayonza |
| Western Province | ~30% | Karongi, Nyamasheke |

> Source: NISR / ICF. *Rwanda Demographic and Health Survey 2019–20.* Kigali 2021.  
> Table 11.1: Nutritional status of children — stunting by province.

---

## Live Links

| Artefact | URL |
|----------|-----|
| **Interactive dashboard** | https://huggingface.co/spaces/Nyingi101/stunting-risk-heatmap |
| **Model (scorer.pkl)** | https://huggingface.co/Nyingi101/stunting-risk-scorer |

## 4-Minute Video

> [Watch 4-minute walkthrough video](https://youtu.be/znyDiG5o8ow)

| Segment | Time | Content |
|---------|------|---------|
| On-camera intro | 0:00–0:30 | Name · Challenge ID · single biggest driver found (sanitation) |
| Code walkthrough | 0:30–1:30 | Live screen-share of `risk_scorer.py::score(household)` |
| Dashboard demo | 1:30–2:30 | `streamlit run dashboard.py` — threshold slider live |
| PDF walkthrough | 2:30–3:30 | Sector Kibilizi PDF briefed as if to a chief |
| 3 questions | 3:30–4:00 | Choropleth gap / A4 layout / NISR baselines |

---

## Submission Checklist

- [x] Public GitHub repository (code + generator + data)
- [x] `households.csv`, `gold_stunting_flag.csv`, GeoJSONs in `data/`
- [x] `risk_scorer.py` with `score(household)` public API
- [x] `dashboard.py` — `streamlit run dashboard.py` launches interactive dashboard
- [x] `printable/` — 15 A4 PDFs (one per sector, bilingual, anonymised)
- [x] `process_log.md` at repo root
- [x] `SIGNED.md` with full name, date, and honor code
- [x] `LICENSE` (MIT)
- [x] 4-minute video URL in README
- [x] `analysis.py` — reproduce all figures in this README with `python analysis.py`

---

## Process Log

See [process_log.md](process_log.md).

## Honor Code

See [SIGNED.md](SIGNED.md).
