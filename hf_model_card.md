---
language:
- en
- rw
license: mit
tags:
- tabular-classification
- public-health
- rwanda
- stunting
- logistic-regression
metrics:
- roc_auc
- f1
model-index:
- name: stunting-risk-scorer
  results:
  - task:
      type: tabular-classification
      name: Tabular Classification
    dataset:
      name: NISR-style synthetic household data
      type: synthetic
    metrics:
    - type: roc_auc
      value: 0.935
    - type: f1
      value: 0.853
---

# stunting-risk-scorer

**S2.T1.2 · AIMS KTT Hackathon · Stunting Risk Heatmap Dashboard**
**Author: Joseph Nyingi Wambua**

## Model description

A logistic regression classifier (scikit-learn `CalibratedClassifierCV` with
Platt scaling, 5-fold cross-validation) that scores each household's stunting
risk on a continuous scale [0, 1].

**Inputs (5 features):**

| Feature | Encoding |
|---------|---------|
| `water_source` | piped=0.0 → river_lake=1.0 |
| `sanitation_tier` | improved=0.0 → open_defecation=1.0 |
| `income_band` | high=0.0 → very_low=1.0 |
| `avg_meal_count` | inverted: 5 meals=0.0 → 1 meal=1.0 |
| `children_under5` | min(n/5, 1.0) |

**Output:** `risk_score` in [0, 1]. Threshold ≥ 0.50 = high risk.

## Performance (held-out 20% test set, n=60)

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.935** |
| Precision | 0.839 |
| Recall | 0.867 |
| F1 | 0.853 |

## Feature importance (LR coefficients, standardised)

| Feature | LR Coefficient |
|---------|---------------|
| Sanitation access | +1.666 |
| Water source quality | +1.536 |
| Income level | +1.070 |
| Meal frequency | +0.801 |
| Children under 5 | +0.516 |

## Intended use

Monthly screening tool for Rwandan community health workers and Umudugudu
chiefs. Outputs feed a printed A4 'sector page' listing the top-10 highest-risk
households with anonymised IDs and one-line intervention hints.

**Not** intended for direct clinical diagnosis. Scores are risk indicators based
on household socioeconomic factors, not measured height-for-age z-scores.

## Training data

Synthetic NISR-style data generated with `generate_data.py` (seed 42).
- 2,500 households across 5 Rwandan districts (Nyarugenge, Gasabo, Kicukiro,
  Nyanza, Musanze)
- 300 gold-labelled households (150 positive, 150 negative)
- Features sampled conditional on district stunting baselines (NISR DHS 2019–20)

## How to use

```python
import joblib
import pandas as pd

# Load the model
model_data = joblib.load("scorer.pkl")
lr, scaler = model_data["lr"], model_data["scaler"]

# Featurize a household
WATER_RISK  = {"piped": 0.0, "protected_well": 0.33,
               "unprotected_well": 0.67, "river_lake": 1.0}
SANIT_RISK  = {"improved": 0.0, "basic": 0.33,
               "limited": 0.67, "open_defecation": 1.0}
INCOME_RISK = {"high": 0.0, "medium": 0.33, "low": 0.67, "very_low": 1.0}

def featurize(row):
    import numpy as np
    return np.array([
        WATER_RISK.get(row["water_source"], 0.5),
        SANIT_RISK.get(row["sanitation_tier"], 0.5),
        INCOME_RISK.get(row["income_band"], 0.5),
        1.0 - (float(row["avg_meal_count"]) - 1.0) / 4.0,
        min(int(row["children_under5"]) / 5.0, 1.0),
    ])

household = {
    "water_source": "unprotected_well",
    "sanitation_tier": "limited",
    "income_band": "low",
    "avg_meal_count": 2,
    "children_under5": 3,
}
feats = featurize(household).reshape(1, -1)
risk_score = lr.predict_proba(scaler.transform(feats))[0, 1]
print(f"Risk score: {risk_score:.3f}  →  {'HIGH RISK' if risk_score >= 0.5 else 'low risk'}")
```

## Limitations

- Trained on synthetic data — not validated against real household surveys
- 300 gold labels is a small training set; real deployment requires DHS or
  CRVS ground-truth labels
- Threshold (0.50) should be re-calibrated for the specific operational context
  (recall vs. precision trade-off depends on health worker capacity)

## Citation

```
@misc{nyingi2026stunting,
  author = {Nyingi, Joseph Wambua},
  title  = {S2.T1.2 Stunting Risk Heatmap Dashboard},
  year   = {2026},
  url    = {https://huggingface.co/Nyingi101/stunting-risk-scorer}
}
```
