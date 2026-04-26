"""
dashboard.py  —  S2.T1.2 Stunting Risk Heatmap Dashboard
Run:  streamlit run dashboard.py
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA_DIR = Path("data")
OUT_DIR  = Path("output")

st.set_page_config(
    page_title="Stunting Risk Dashboard · Rwanda",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Feature encoding (mirrors risk_scorer.py) ────────────────────────────────
WATER_RISK  = {"piped": 0.00, "protected_well": 0.33,
               "unprotected_well": 0.67, "river_lake": 1.00}
SANIT_RISK  = {"improved": 0.00, "basic": 0.33,
               "limited": 0.67, "open_defecation": 1.00}
INCOME_RISK = {"high": 0.00, "medium": 0.33, "low": 0.67, "very_low": 1.00}
INCOME_UP   = {"very_low": "low", "low": "medium",
               "medium": "high", "high": "high"}
RULE_WEIGHTS = np.array([0.30, 0.25, 0.25, 0.12, 0.08])

def featurize_batch(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["water_source"].map(WATER_RISK).fillna(0.5),
        df["sanitation_tier"].map(SANIT_RISK).fillna(0.5),
        df["income_band"].map(INCOME_RISK).fillna(0.5),
        1.0 - (df["avg_meal_count"].astype(float) - 1.0) / 4.0,
        (df["children_under5"].astype(int) / 5.0).clip(upper=1.0),
    ])

# ── Data (cached) ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hh     = pd.read_csv(OUT_DIR / "households_scored.csv")
    hh_raw = pd.read_csv(DATA_DIR / "households.csv")
    sector = pd.read_csv(OUT_DIR / "sector_summary.csv")
    with open(DATA_DIR / "sectors.geojson") as f:
        sectors_geo = json.load(f)
    with open(OUT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return hh, hh_raw, sector, sectors_geo, metrics

@st.cache_resource
def load_scorer():
    data = joblib.load(OUT_DIR / "scorer.pkl")
    return data["lr"], data["scaler"]

hh, hh_raw, sector_summary, sectors_geo, metrics = load_data()
lr_model, scaler = load_scorer()

def rescore(df_raw: pd.DataFrame) -> pd.Series:
    X = featurize_batch(df_raw)
    probs = lr_model.predict_proba(scaler.transform(X))[:, 1]
    return pd.Series(probs, index=df_raw.index)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Stunting Risk")
    st.caption("S2.T1.2 · AIMS KTT Hackathon")
    st.caption("Author: Joseph Nyingi Wambua")
    st.divider()

    all_districts  = sorted(hh["district"].unique())
    sel_districts  = st.multiselect("District filter", all_districts, default=all_districts)
    risk_threshold = st.slider("Risk threshold", 0.0, 1.0, 0.50, 0.05,
                                help="Households at or above this score are flagged high-risk")
    show_points    = st.toggle("Show household points on map", value=True)

    st.divider()
    st.caption("Model performance (held-out 20% test, n=60)")
    c1, c2 = st.columns(2)
    c1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    c1.metric("Recall",  f"{metrics['recall']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c2.metric("F1",        f"{metrics['f1']:.3f}")
    st.divider()
    st.caption("**Biggest driver:** Sanitation access (coeff 1.67)")
    st.caption("**NISR baselines:** Kigali ~20%, Northern ~40%, Southern ~38%")

# ── Filter ───────────────────────────────────────────────────────────────────
filtered  = hh[hh["district"].isin(sel_districts)]
at_risk   = filtered[filtered["risk_score"] >= risk_threshold]
sec_filt  = sector_summary[sector_summary["district"].isin(sel_districts)]
raw_filt  = hh_raw[hh_raw["district"].isin(sel_districts)]

# ── KPI row ──────────────────────────────────────────────────────────────────
st.title("Rwanda Childhood Stunting Risk Heatmap")
st.caption(
    "AIMS KTT Hackathon S2.T1.2 · Synthetic NISR-style data · "
    "2,500 households · 5 districts · 15 sectors · "
    "[GitHub](https://github.com/Josephnyingi/stunting-risk-dashboard) · "
    "[HF Space](https://huggingface.co/spaces/Nyingi101/stunting-risk-heatmap)"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Households shown", f"{len(filtered):,}")
c2.metric("High-risk",        f"{len(at_risk):,}")
c3.metric("Prevalence",       f"{len(at_risk)/max(len(filtered),1):.1%}")
c4.metric("Most at-risk sector",
          sec_filt.loc[sec_filt["pct_high_risk"].idxmax(), "sector"]
          if len(sec_filt) else "—")
c5.metric("Avg risk score",   f"{filtered['risk_score'].mean():.3f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_sector, tab_table, tab_drivers, tab_whatif = st.tabs([
    "🗺 Choropleth Map", "📊 Sector Analysis",
    "📋 Household Table", "🔑 Risk Drivers", "🔮 What-If Simulator",
])

# ═══════════════ TAB 1 : CHOROPLETH MAP ═════════════════════════════════════
with tab_map:
    col_map, col_bar = st.columns([3, 1])

    with col_map:
        fig_choro = px.choropleth_map(
            sec_filt,
            geojson=sectors_geo,
            locations="sector",
            featureidkey="properties.sector",
            color="pct_high_risk",
            color_continuous_scale="YlOrRd",
            range_color=(0, max(sec_filt["pct_high_risk"].max() + 0.05, 0.1)),
            map_style="carto-positron",
            zoom=7.2,
            center={"lat": -1.95, "lon": 29.90},
            opacity=0.70,
            hover_data={"district": True, "sector": True,
                        "pct_high_risk": ":.1%", "n_households": True},
            labels={"pct_high_risk": "% High-risk", "n_households": "Households"},
            title=f"Sector-level stunting risk  (threshold ≥ {risk_threshold:.2f})",
            height=540,
        )
        if show_points and len(at_risk) <= 600:
            fig_choro.add_trace(go.Scattermapbox(
                lat=at_risk["lat"], lon=at_risk["lon"],
                mode="markers",
                marker=dict(size=5, color="crimson", opacity=0.55),
                text=at_risk.apply(
                    lambda r: f"ID: {r['household_id']}<br>"
                              f"Score: {r['risk_score']:.2f}<br>"
                              f"Sector: {r['sector']}", axis=1),
                hoverinfo="text",
                name="High-risk household",
            ))
        fig_choro.update_layout(margin={"r": 0, "l": 0, "t": 40, "b": 0},
                                coloraxis_colorbar_title="% High-risk")
        st.plotly_chart(fig_choro, use_container_width=True)

        # ── Offline HTML export ───────────────────────────────────────────────
        st.divider()
        st.markdown("**📥 Offline HTML Export** — download the map as a self-contained file (no internet needed to open)")
        if st.button("Generate offline HTML report"):
            with st.spinner("Building offline report…"):
                html = _build_offline_html(fig_choro, sec_filt, at_risk, risk_threshold, sel_districts)
            st.download_button(
                label="⬇ Download offline_report.html",
                data=html,
                file_name="stunting_risk_offline_report.html",
                mime="text/html",
            )
            st.success("Ready! Open the downloaded file on any laptop — no internet required.")

    with col_bar:
        dist_agg = (
            filtered.groupby("district")["risk_score"]
            .apply(lambda x: (x >= risk_threshold).mean())
            .reset_index()
            .rename(columns={"risk_score": "pct_high_risk"})
            .sort_values("pct_high_risk", ascending=True)
        )
        fig_bar = px.bar(
            dist_agg, x="pct_high_risk", y="district", orientation="h",
            color="pct_high_risk", color_continuous_scale="YlOrRd",
            labels={"pct_high_risk": "% High-risk", "district": ""},
            title="By district", height=540,
        )
        fig_bar.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin={"r": 10, "l": 0, "t": 40, "b": 0})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.info(
        "**Reading the map:** Red = high % of households above the risk threshold. "
        "Rural districts (Nyanza, Musanze) score higher due to river-water use and "
        "open defecation. NISR baselines: Kigali ~20%, Northern ~40%, Southern ~38%."
    )

# ═══════════════ TAB 2 : SECTOR ANALYSIS ════════════════════════════════════
with tab_sector:
    st.subheader("Sector-level risk summary (all 15 sectors)")
    st.dataframe(
        sec_filt.sort_values("pct_high_risk", ascending=False)
        .style.background_gradient(subset=["pct_high_risk", "avg_risk_score"], cmap="YlOrRd"),
        use_container_width=True,
    )
    fig_scatter = px.scatter(
        sec_filt, x="avg_risk_score", y="pct_high_risk",
        size="n_households", color="district", text="sector",
        labels={"avg_risk_score": "Avg risk score", "pct_high_risk": "% High-risk"},
        title="Sectors: average score vs prevalence (bubble = #households)",
        height=420,
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════ TAB 3 : HOUSEHOLD TABLE ════════════════════════════════════
with tab_table:
    COLS = ["household_id", "district", "sector", "risk_score", "risk_tier",
            "children_under5", "avg_meal_count", "water_source",
            "sanitation_tier", "income_band", "urban_rural", "intervention"]
    st.subheader(f"High-risk households  (score ≥ {risk_threshold:.2f})")
    st.dataframe(
        at_risk[COLS].sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
        .style.background_gradient(subset=["risk_score"], cmap="YlOrRd"),
        use_container_width=True, height=480,
    )
    csv_bytes = at_risk[COLS].to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv_bytes,
                       "high_risk_households.csv", "text/csv")

# ═══════════════ TAB 4 : RISK DRIVERS ═══════════════════════════════════════
with tab_drivers:
    col_feat, col_hist = st.columns(2)
    with col_feat:
        feat_df = pd.DataFrame({
            "Feature": ["Sanitation access", "Water source",
                        "Income level", "Meal frequency", "Children <5"],
            "LR Coefficient": [1.666, 1.536, 1.070, 0.801, 0.516],
        }).sort_values("LR Coefficient", ascending=True)
        fig_feat = px.bar(
            feat_df, x="LR Coefficient", y="Feature", orientation="h",
            color="LR Coefficient", color_continuous_scale="RdYlGn_r",
            title="Feature importance — LR coefficients (standardised)", height=320,
        )
        fig_feat.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.caption("Sanitation (1.67) edges water source (1.54) as top LR driver.")
    with col_hist:
        fig_hist = px.histogram(
            filtered, x="risk_score", nbins=40,
            color="district", barmode="overlay", opacity=0.65,
            title="Risk score distribution by district", height=320,
        )
        fig_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold {risk_threshold:.2f}")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Water source × risk tier")
    water_ct = pd.crosstab(
        filtered["water_source"], filtered["risk_tier"], normalize="index"
    ).round(3) * 100
    fig_w = px.bar(
        water_ct.reset_index().melt(id_vars="water_source",
                                     var_name="risk_tier", value_name="pct"),
        x="water_source", y="pct", color="risk_tier",
        color_discrete_map={"critical": "#b22222", "high": "#e05c00",
                            "moderate": "#e6a817", "low": "#4a7c59"},
        labels={"pct": "% of water-source group", "water_source": "Water source"},
        title="River/lake users are 87.6% critical vs 8% for piped-water users",
        height=340, barmode="stack",
    )
    st.plotly_chart(fig_w, use_container_width=True)

# ═══════════════ TAB 5 : WHAT-IF SIMULATOR ══════════════════════════════════
with tab_whatif:
    st.subheader("🔮 Counterfactual Intervention Simulator")
    st.markdown(
        "Choose an intervention and see how the sector-level risk profile changes. "
        "Scores are recomputed live using the trained logistic regression model."
    )

    INTERVENTIONS = {
        "💧 Provide piped water to all households": {
            "field": "water_source", "value": "piped",
            "scope": "all", "cost_usd": 150,
            "label": "piped water connection",
        },
        "🚽 Upgrade sanitation to improved for all": {
            "field": "sanitation_tier", "value": "improved",
            "scope": "all", "cost_usd": 80,
            "label": "improved latrine (VIP/pour-flush)",
        },
        "💰 Income support — raise 1 band (high-risk only)": {
            "field": "income_band", "value": "__raise__",
            "scope": "high_risk", "cost_usd": 50,
            "label": "monthly Ubudehe cash transfer",
        },
        "🍽 Nutrition: +1 meal/day (high-risk only)": {
            "field": "avg_meal_count", "value": "__plus1__",
            "scope": "high_risk", "cost_usd": 30,
            "label": "supplementary feeding (RUTF/Imbuto)",
        },
    }

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        chosen = st.selectbox("Select intervention", list(INTERVENTIONS.keys()))
        iv = INTERVENTIONS[chosen]
        st.markdown(f"**Cost proxy:** ~${iv['cost_usd']:,} USD per household  \n"
                    f"*(WASH/NISR reference estimate)*")

        scope_label = ("All households in filtered districts"
                       if iv["scope"] == "all"
                       else "High-risk households only (score ≥ threshold)")
        st.caption(f"Applied to: {scope_label}")
        run_sim = st.button("▶ Run simulation", type="primary")

    if run_sim:
        with col_result:
            with st.spinner("Rescoring…"):
                # Build modified raw dataframe
                df_cf = raw_filt.copy()

                if iv["scope"] == "high_risk":
                    # Identify current high-risk IDs
                    hr_ids = set(at_risk["household_id"])
                    mask = df_cf["household_id"].isin(hr_ids)
                else:
                    mask = pd.Series(True, index=df_cf.index)

                if iv["value"] == "__raise__":
                    df_cf.loc[mask, "income_band"] = (
                        df_cf.loc[mask, "income_band"].map(INCOME_UP)
                    )
                elif iv["value"] == "__plus1__":
                    df_cf.loc[mask, "avg_meal_count"] = (
                        df_cf.loc[mask, "avg_meal_count"].clip(upper=4) + 1
                    )
                else:
                    df_cf.loc[mask, iv["field"]] = iv["value"]

                df_cf["risk_score_new"] = rescore(df_cf)

                # Sector-level before/after
                before = (
                    filtered.groupby("sector")["risk_score"]
                    .apply(lambda x: (x >= risk_threshold).mean())
                    .rename("before")
                )
                after_df = df_cf.merge(
                    hh[["household_id", "sector"]], on="household_id"
                )
                after = (
                    after_df.groupby("sector")["risk_score_new"]
                    .apply(lambda x: (x >= risk_threshold).mean())
                    .rename("after")
                )
                comp = pd.concat([before, after], axis=1).dropna().reset_index()
                comp["reduction_pp"] = (comp["before"] - comp["after"]) * 100
                comp = comp.sort_values("reduction_pp", ascending=False)

                # KPIs
                n_before = len(at_risk[at_risk["district"].isin(sel_districts)])
                n_after  = int((df_cf["risk_score_new"] >= risk_threshold).sum())
                graduates = n_before - n_after
                n_treated = int(mask.sum())
                total_cost = n_treated * iv["cost_usd"]

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("High-risk before", f"{n_before:,}")
            k2.metric("High-risk after",  f"{n_after:,}",
                      delta=f"-{graduates:,}", delta_color="inverse")
            k3.metric("Households treated", f"{n_treated:,}")
            k4.metric("Est. total cost",    f"${total_cost:,.0f}")

            # Before/after bar chart
            fig_comp = go.Figure()
            fig_comp.add_bar(name="Before", x=comp["sector"],
                             y=comp["before"] * 100,
                             marker_color="#d62728")
            fig_comp.add_bar(name="After intervention", x=comp["sector"],
                             y=comp["after"] * 100,
                             marker_color="#4a7c59")
            fig_comp.update_layout(
                barmode="group",
                title=f"% High-risk per sector — before vs after: {chosen}",
                yaxis_title="% High-risk households",
                xaxis_title="Sector",
                height=380,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Reduction table
            st.markdown("**Reduction by sector (percentage points)**")
            st.dataframe(
                comp[["sector", "before", "after", "reduction_pp"]]
                .rename(columns={"before": "% Before", "after": "% After",
                                  "reduction_pp": "Reduction (pp)"})
                .style.format({"% Before": "{:.1%}", "% After": "{:.1%}",
                               "Reduction (pp)": "{:.1f}"})
                .background_gradient(subset=["Reduction (pp)"], cmap="Greens"),
                use_container_width=True,
            )

            # Graduated households
            graduated_hh = df_cf[
                (df_cf["household_id"].isin(at_risk["household_id"])) &
                (df_cf["risk_score_new"] < risk_threshold)
            ].merge(hh[["household_id", "sector", "district"]], on="household_id")
            if len(graduated_hh):
                st.success(
                    f"✅ **{len(graduated_hh)} households graduate from high-risk to safe** "
                    f"under this intervention."
                )
    else:
        with col_result:
            st.info("Select an intervention and press **▶ Run simulation** to see the impact.")


# ═══════════════ OFFLINE HTML BUILDER ═══════════════════════════════════════
def _build_offline_html(fig_choro, sec_filt, at_risk, threshold, districts):
    """Build a fully self-contained HTML report (no internet needed to open)."""
    choro_html = pio.to_html(fig_choro, full_html=False, include_plotlyjs="cdn")

    sector_rows = "".join(
        f"<tr><td>{r.district}</td><td>{r.sector}</td>"
        f"<td>{r.n_households}</td>"
        f"<td style='background:hsl({int(120*(1-r.pct_high_risk))},60%,70%)'>"
        f"{r.pct_high_risk:.1%}</td>"
        f"<td>{r.avg_risk_score:.3f}</td></tr>"
        for r in sec_filt.sort_values("pct_high_risk", ascending=False).itertuples()
    )
    hh_rows = "".join(
        f"<tr><td>{r.household_id}</td><td>{r.district}</td><td>{r.sector}</td>"
        f"<td style='background:hsl({int(120*(1-r.risk_score))},60%,70%)'>"
        f"{r.risk_score:.3f}</td>"
        f"<td>{r.risk_tier}</td><td>{r.water_source}</td>"
        f"<td>{r.sanitation_tier}</td><td>{r.income_band}</td>"
        f"<td>{r.intervention}</td></tr>"
        for r in at_risk.sort_values("risk_score", ascending=False).head(100).itertuples()
    )
    from datetime import date
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rwanda Stunting Risk — Offline Report</title>
<style>
  body{{font-family:Arial,sans-serif;margin:24px;color:#222;background:#fafafa}}
  h1{{color:#8b0000}}h2{{color:#444;margin-top:32px}}
  .kpi{{display:flex;gap:16px;margin:16px 0}}
  .kpi-box{{background:#fff;border:1px solid #ddd;border-radius:8px;
            padding:16px 24px;text-align:center;min-width:130px}}
  .kpi-box .val{{font-size:2em;font-weight:bold;color:#8b0000}}
  .kpi-box .lbl{{font-size:.8em;color:#666}}
  table{{border-collapse:collapse;width:100%;font-size:.85em;margin-top:8px}}
  th{{background:#8b0000;color:#fff;padding:6px 10px;text-align:left}}
  td{{padding:5px 10px;border-bottom:1px solid #eee}}
  tr:hover td{{background:#fff3f3}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;
          font-size:.75em;font-weight:bold;color:#fff;background:#8b0000}}
  footer{{margin-top:40px;font-size:.75em;color:#999;border-top:1px solid #ddd;padding-top:12px}}
</style>
</head>
<body>
<h1>🏥 Rwanda Childhood Stunting Risk — Offline Report</h1>
<p><strong>Generated:</strong> {date.today().strftime('%d %B %Y')} &nbsp;|&nbsp;
   <strong>Districts:</strong> {', '.join(districts)} &nbsp;|&nbsp;
   <strong>Threshold:</strong> ≥{threshold:.2f} &nbsp;|&nbsp;
   <span class="badge">OFFLINE</span></p>

<div class="kpi">
  <div class="kpi-box"><div class="val">{len(at_risk):,}</div><div class="lbl">High-risk households</div></div>
  <div class="kpi-box"><div class="val">{len(at_risk)/max(len(sec_filt),1)*len(sec_filt):.0f}</div><div class="lbl">Sectors shown</div></div>
  <div class="kpi-box"><div class="val">{at_risk['risk_score'].mean():.3f}</div><div class="lbl">Avg risk score</div></div>
  <div class="kpi-box"><div class="val">{sec_filt.loc[sec_filt['pct_high_risk'].idxmax(),'sector']}</div><div class="lbl">Most at-risk sector</div></div>
</div>

<h2>Choropleth Map</h2>
{choro_html}

<h2>Sector Summary</h2>
<table>
<tr><th>District</th><th>Sector</th><th>Households</th>
    <th>% High-risk</th><th>Avg score</th></tr>
{sector_rows}
</table>

<h2>Top 100 High-Risk Households</h2>
<table>
<tr><th>ID</th><th>District</th><th>Sector</th><th>Score</th><th>Tier</th>
    <th>Water</th><th>Sanitation</th><th>Income</th><th>Intervention</th></tr>
{hh_rows}
</table>

<footer>
  S2.T1.2 · AIMS KTT Hackathon · Author: Joseph Nyingi Wambua ·
  Model: Nyingi101/stunting-risk-scorer ·
  Data: NISR-style synthetic (seed 42)
</footer>
</body></html>"""
