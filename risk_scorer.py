"""
risk_scorer.py
Per-household stunting risk scorer: logistic regression trained on gold labels
with a rule-based fallback when no model is fitted.

Public API (used in video demo):
    from risk_scorer import score
    score(household_row)  →  float in [0, 1]

CLI:
    python risk_scorer.py   # scores all households, writes output/
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

DATA_DIR = Path("data")
OUT_DIR  = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ── Feature encoding (ordered; index positions are shared with FEATURE_NAMES) ─
WATER_RISK  = {"piped": 0.00, "protected_well": 0.33,
               "unprotected_well": 0.67, "river_lake": 1.00}
SANIT_RISK  = {"improved": 0.00, "basic": 0.33,
               "limited": 0.67, "open_defecation": 1.00}
INCOME_RISK = {"high": 0.00, "medium": 0.33, "low": 0.67, "very_low": 1.00}

FEATURE_NAMES = [
    "water_risk", "sanit_risk", "income_risk", "meal_norm", "children_norm"
]
RULE_WEIGHTS = np.array([0.30, 0.25, 0.25, 0.12, 0.08])

DRIVER_LABELS = {
    "water_risk":    "Water source quality",
    "sanit_risk":    "Sanitation access",
    "income_risk":   "Income level",
    "meal_norm":     "Meal frequency (low)",
    "children_norm": "Number of children under 5",
}
INTERVENTION_MAP = {
    "water_risk":    "WASH upgrade — connect to protected/piped water source",
    "sanit_risk":    "Sanitation — install improved latrine (VIP/pour-flush)",
    "income_risk":   "Social protection — refer to Ubudehe / cash-transfer programme",
    "meal_norm":     "Nutrition — enroll in supplementary feeding (RUTF / Imbuto)",
    "children_norm": "Referral — CHW multi-child nutrition screening",
}


def featurize(row: pd.Series) -> np.ndarray:
    """Map a household row to a 5-dimensional [0,1] feature vector."""
    return np.array([
        WATER_RISK.get(str(row["water_source"]), 0.50),
        SANIT_RISK.get(str(row["sanitation_tier"]), 0.50),
        INCOME_RISK.get(str(row["income_band"]), 0.50),
        1.0 - (float(row["avg_meal_count"]) - 1.0) / 4.0,   # meals 1–5 → 1.0–0.0
        min(int(row["children_under5"]) / 5.0, 1.0),
    ], dtype=float)


def rule_score(row: pd.Series) -> float:
    """Weighted rule-based risk score in [0, 1]. No ML required."""
    return float(np.clip(np.dot(featurize(row), RULE_WEIGHTS), 0.0, 1.0))


def top_drivers(row: pd.Series, n: int = 3) -> list:
    """Return human-readable labels for the top-n risk-contributing features."""
    contribs = featurize(row) * RULE_WEIGHTS
    top_idx  = np.argsort(contribs)[::-1][:n]
    return [DRIVER_LABELS[FEATURE_NAMES[i]] for i in top_idx]


def top_intervention(row: pd.Series) -> str:
    """Return the single highest-priority intervention hint."""
    contribs = featurize(row) * RULE_WEIGHTS
    return INTERVENTION_MAP[FEATURE_NAMES[int(np.argmax(contribs))]]


class RiskScorer:
    """
    Hybrid scorer: logistic regression calibrated on gold labels.
    Falls back to rule_score() if not yet fitted.

    Threshold calibration: default 0.50 (≥ 0.50 = high risk).
    Calibrated from LR probability output; gold-label AUC-ROC is reported.
    """

    def __init__(self):
        self.lr      = CalibratedClassifierCV(
                           LogisticRegression(C=1.0, max_iter=500, random_state=42),
                           cv=5, method="sigmoid"
                       )
        self.scaler  = StandardScaler()
        self.fitted  = False

    def fit(self, households: pd.DataFrame, gold: pd.DataFrame) -> dict:
        merged = households.merge(gold, on="household_id")
        X = np.vstack(merged.apply(featurize, axis=1))
        y = merged["stunting_flag"].values

        # 80/20 stratified split — metrics evaluated on held-out test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        self.lr.fit(X_train_sc, y_train)
        self.fitted = True

        # Honest metrics on held-out test set
        probs = self.lr.predict_proba(X_test_sc)[:, 1]
        preds = (probs >= 0.50).astype(int)
        auc   = roc_auc_score(y_test, probs)
        rep   = classification_report(y_test, preds, output_dict=True)

        metrics = {
            "auc_roc":   round(auc, 4),
            "precision": round(rep["1"]["precision"], 4),
            "recall":    round(rep["1"]["recall"], 4),
            "f1":        round(rep["1"]["f1-score"], 4),
            "n_train":   int(len(X_train)),
            "n_test":    int(len(X_test)),
        }
        return metrics

    def score(self, row: pd.Series) -> float:
        """
        Score a single household row.
        Returns risk probability in [0, 1].
        Threshold ≥ 0.50 = high risk.
        """
        feats = featurize(row).reshape(1, -1)
        if self.fitted:
            return float(self.lr.predict_proba(self.scaler.transform(feats))[0, 1])
        return rule_score(row)

    def score_batch(self, df: pd.DataFrame) -> pd.Series:
        X = np.vstack(df.apply(featurize, axis=1))
        if self.fitted:
            probs = self.lr.predict_proba(self.scaler.transform(X))[:, 1]
            return pd.Series(probs, index=df.index)
        return pd.Series(np.clip(X @ RULE_WEIGHTS, 0, 1), index=df.index)

    def save(self, path: str = "output/scorer.pkl"):
        joblib.dump({"lr": self.lr, "scaler": self.scaler, "fitted": self.fitted}, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "output/scorer.pkl") -> "RiskScorer":
        obj  = cls()
        data = joblib.load(path)
        obj.lr, obj.scaler, obj.fitted = data["lr"], data["scaler"], data["fitted"]
        return obj


# ── Module-level singleton (lazy-loaded for `from risk_scorer import score`) ─
_scorer: RiskScorer | None = None

def score(household: pd.Series) -> float:
    """
    Public API — score a single household Series.
    Auto-loads the trained model from output/scorer.pkl if available.
    """
    global _scorer
    if _scorer is None:
        model_path = OUT_DIR / "scorer.pkl"
        _scorer = RiskScorer.load(str(model_path)) if model_path.exists() else RiskScorer()
    return _scorer.score(household)


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data …")
    households = pd.read_csv(DATA_DIR / "households.csv")
    gold       = pd.read_csv(DATA_DIR / "gold_stunting_flag.csv")

    scorer  = RiskScorer()
    print("Training logistic regression on gold labels …")
    metrics = scorer.fit(households, gold)

    print("\n── Model metrics ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")

    print("\nScoring 2,500 households …")
    households["risk_score"] = scorer.score_batch(households)
    households["risk_tier"]  = pd.cut(
        households["risk_score"],
        bins=[0, 0.35, 0.55, 0.75, 1.01],
        labels=["low", "moderate", "high", "critical"],
        right=False,
    )
    households["top_drivers"]  = households.apply(
        lambda r: " | ".join(top_drivers(r, n=3)), axis=1
    )
    households["intervention"] = households.apply(top_intervention, axis=1)

    # Sector-level aggregation
    sector_summary = (
        households.groupby(["district", "sector"])
        .agg(
            n_households   = ("household_id", "count"),
            avg_risk_score = ("risk_score", "mean"),
            pct_high_risk  = ("risk_score", lambda x: (x >= 0.50).mean()),
        )
        .reset_index()
    )
    sector_summary[["avg_risk_score", "pct_high_risk"]] = \
        sector_summary[["avg_risk_score", "pct_high_risk"]].round(4)

    households.to_csv(OUT_DIR / "households_scored.csv", index=False)
    sector_summary.to_csv(OUT_DIR / "sector_summary.csv", index=False)
    scorer.save()

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFiles written to {OUT_DIR}/")
    print(f"  households_scored.csv : {len(households)} rows")
    print(f"  sector_summary.csv    : {len(sector_summary)} sectors")

    print("\nTop 5 highest-risk sectors:")
    top = sector_summary.sort_values("pct_high_risk", ascending=False).head(5)
    print(top[["district", "sector", "pct_high_risk", "avg_risk_score"]].to_string(index=False))
