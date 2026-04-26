---
title: Stunting Risk Heatmap Dashboard
emoji: 🏥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# S2.T1.2 · Stunting Risk Heatmap Dashboard

**AIMS KTT Hackathon · Tier 1 · HealthTech / Geospatial / Data Viz**
**Author: Joseph Nyingi Wambua**

Interactive dashboard for identifying childhood stunting risk across 5 Rwandan districts
and 15 sectors, using synthetic NISR-style household data.

## Features

- **Choropleth map** — sector-level risk, colour-coded YlOrRd
- **Risk threshold slider** — real-time filtering of high-risk households
- **District filter** — isolate any combination of the 5 districts
- **Household table** — sortable, downloadable CSV of flagged households
- **Risk driver charts** — feature importance and water-source breakdown

## Model

Logistic regression with Platt calibration trained on 300 gold-labelled households
(AUC-ROC 0.935 on held-out 20% test set).

Biggest driver: **sanitation access** (LR coeff 1.67), followed by water source (1.54).

## Full submission

GitHub: [Nyingi101/stunting-risk-dashboard](https://github.com/Nyingi101/stunting-risk-dashboard)
Model: [Nyingi101/stunting-risk-scorer](https://huggingface.co/Nyingi101/stunting-risk-scorer)
