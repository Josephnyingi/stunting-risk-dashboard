# process_log.md — S2.T1.2 Stunting Risk Heatmap Dashboard

## Hour-by-hour timeline

| Time    | Activity |
|---------|----------|
| H+0:00  | Read brief. Identified scoring rubric: Product & Business Adaptation weighted equally with Technical and Model (20% each). Decided to invest heavily in paper workflow design. |
| H+0:20  | Designed data generator: 5 districts with NISR-realistic stunting baselines. Chose bounding-box sector polygons for reliable choropleth rendering. |
| H+0:45  | Wrote `generate_data.py`. Key decision: use `np.random.default_rng(42)` throughout for full reproducibility, not `random.seed()`. |
| H+1:10  | Wrote `risk_scorer.py`. Chose logistic regression over random forest: interpretable weights are defensible in the live defense and directly answer "which driver matters most." |
| H+1:40  | Wrote `dashboard.py`. Used Plotly `choropleth_mapbox` instead of `streamlit-folium` — fewer dependencies, faster render, natively cached by Streamlit. |
| H+2:10  | Wrote `export_printable.py`. Invested time in: bilingual labels (English + Kinyarwanda), anonymisation logic (sector code + rank, no household ID on paper), annotation block for paper workflow. |
| H+2:40  | Ran full pipeline: generate → score → dashboard → PDFs. Fixed off-by-one in `pd.cut` bins (right=False for [0.75, 1.0) tier). |
| H+3:00  | Wrote README including NISR real-data references and district meeting loop table. |
| H+3:20  | Tested Streamlit: threshold slider + district multiselect + scatter overlay + download button. |
| H+3:40  | Wrote process_log.md, SIGNED.md. Final review of submission checklist. |

---

## LLM / tool use

| Tool | Why used |
|------|----------|
| Claude Code (claude-sonnet-4-6) | Code generation scaffolding and iteration |

---

## Three sample prompts I actually sent

1. **Prompt sent:**  
   *"Write a `featurize(row)` function that maps a household pandas Series to a 5-element numpy array using the water/sanitation/income/meal/children features, scaled to [0,1]."*  
   → Used because encoding lookup tables are error-prone to write manually; reviewed output against brief feature list before keeping.

2. **Prompt sent:**  
   *"Write a ReportLab A4 PDF that shows a top-10 table with risk-score colored cells, a bilingual English/Kinyarwanda header, and an annotation block at the bottom for a paper workflow."*  
   → Used for ReportLab table style syntax (TableStyle indices are easy to get wrong). Manually adjusted column widths and font sizes after reviewing render.

3. **Prompt sent:**  
   *"Generate Plotly choropleth_mapbox code using a GeoJSON featureidkey on 'properties.sector', with a YlOrRd scale and carto-positron tiles."*  
   → Used to get the `featureidkey` syntax right on first try (common source of silent mismatches). Verified by checking that all 15 sectors colored correctly.

**Prompt I discarded and why:**  
*"Predict stunting risk using a random forest with SHAP values for explainability."*  
Discarded because: (a) the brief explicitly says deep ML is not required, (b) SHAP adds 3+ dependencies that slow Colab startup past the 3 s render budget, and (c) logistic regression weights are just as defensible in the live defense and easier to explain to a non-technical evaluator.

---

## The single hardest decision

**Rule-based vs. logistic regression as the primary scorer.**

A pure rule-based scorer is fully transparent and requires no gold labels, but
it cannot report AUC-ROC (a required metric) and risks miscalibrated weights.
A pure LR model trained on 300 gold labels is better calibrated, but if the
gold-label sample is biased the scores mislead community workers.

I resolved this by using LR as the default (when gold labels are present) and
falling back to the rule-based scorer when no model file is found — so the
pipeline degrades gracefully in a new deployment. I calibrated threshold at
0.50 (LR probability midpoint) rather than optimising for recall, because
over-flagging high-risk households erodes community trust faster than missing
a borderline case.
