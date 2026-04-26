"""
app.py  —  S2.T1.2 Stunting Risk Heatmap Dashboard
Hugging Face Spaces (Streamlit) entry point.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"

st.set_page_config(
    page_title="Stunting Risk Dashboard · Rwanda",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    hh     = pd.read_csv(DATA_DIR / "households_scored.csv")
    sector = pd.read_csv(DATA_DIR / "sector_summary.csv")
    with open(DATA_DIR / "sectors.geojson") as f:
        sectors_geo = json.load(f)
    with open(DATA_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return hh, sector, sectors_geo, metrics

hh, sector_summary, sectors_geo, metrics = load_data()

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
    col1, col2 = st.columns(2)
    col1.metric("AUC-ROC",   f"{metrics['auc_roc']:.3f}")
    col1.metric("Recall",    f"{metrics['recall']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("F1",        f"{metrics['f1']:.3f}")

    st.divider()
    st.caption("**Biggest driver:** Sanitation access (LR coeff 1.67)")
    st.caption("**NISR baselines:** Kigali ~20%, Northern ~40%, Southern ~38%")

# ── Filter ───────────────────────────────────────────────────────────────────
filtered  = hh[hh["district"].isin(sel_districts)]
at_risk   = filtered[filtered["risk_score"] >= risk_threshold]
sec_filt  = sector_summary[sector_summary["district"].isin(sel_districts)]

# ── KPI row ──────────────────────────────────────────────────────────────────
st.title("Rwanda Childhood Stunting Risk Heatmap")
st.caption(
    "AIMS KTT Hackathon S2.T1.2 · Synthetic NISR-style data · "
    "2,500 households · 5 districts · 15 sectors · "
    "[GitHub](https://github.com/Nyingi101/stunting-risk-dashboard)"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Households shown",    f"{len(filtered):,}")
c2.metric("High-risk",           f"{len(at_risk):,}")
c3.metric("Prevalence",          f"{len(at_risk)/max(len(filtered),1):.1%}")
c4.metric("Most at-risk sector",
          sec_filt.loc[sec_filt["pct_high_risk"].idxmax(), "sector"]
          if len(sec_filt) else "—")
c5.metric("Avg risk score",      f"{filtered['risk_score'].mean():.3f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_sector, tab_table, tab_drivers = st.tabs(
    ["🗺 Choropleth Map", "📊 Sector Analysis", "📋 Household Table", "🔑 Risk Drivers"]
)

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
        "**Reading the map:** Red = high % of households with risk score ≥ threshold. "
        "Rural districts (Nyanza, Musanze) score higher due to river-water use and "
        "open defecation — not because stunting is directly measured. "
        "NISR baselines: Kigali ~20%, Northern Province ~40%, Southern ~38%."
    )

# ═══════════════ TAB 2 : SECTOR ANALYSIS ════════════════════════════════════
with tab_sector:
    st.subheader("Sector-level risk summary (all 15 sectors)")
    st.dataframe(
        sec_filt.sort_values("pct_high_risk", ascending=False)
        .style.background_gradient(subset=["pct_high_risk", "avg_risk_score"],
                                    cmap="YlOrRd"),
        use_container_width=True,
    )
    fig_scatter = px.scatter(
        sec_filt, x="avg_risk_score", y="pct_high_risk",
        size="n_households", color="district", text="sector",
        labels={"avg_risk_score": "Avg risk score",
                "pct_high_risk": "% High-risk"},
        title="Sectors: average score vs high-risk prevalence (bubble = #households)",
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
            "Rule Weight":    [0.25,  0.30,  0.25,  0.12,  0.08],
        }).sort_values("LR Coefficient", ascending=True)

        fig_feat = px.bar(
            feat_df, x="LR Coefficient", y="Feature", orientation="h",
            color="LR Coefficient", color_continuous_scale="RdYlGn_r",
            title="Feature importance — LR coefficients (standardised)",
            height=320,
        )
        fig_feat.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.caption(
            "Sanitation access (1.67) edges water source (1.54) as the top LR driver. "
            "Both are the dominant risk factors — together >55% of the model weight."
        )

    with col_hist:
        fig_hist = px.histogram(
            filtered, x="risk_score", nbins=40,
            color="district", barmode="overlay", opacity=0.65,
            title="Risk score distribution by district", height=320,
        )
        fig_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold {risk_threshold:.2f}")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Water source × risk tier (key finding)")
    water_ct = pd.crosstab(
        filtered["water_source"], filtered["risk_tier"],
        normalize="index"
    ).round(3) * 100
    fig_w = px.bar(
        water_ct.reset_index().melt(id_vars="water_source",
                                     var_name="risk_tier", value_name="pct"),
        x="water_source", y="pct", color="risk_tier",
        color_discrete_map={
            "critical": "#b22222", "high": "#e05c00",
            "moderate": "#e6a817", "low": "#4a7c59"
        },
        labels={"pct": "% of water-source group", "water_source": "Water source"},
        title="Risk tier breakdown by water source — river/lake users are 87.6% critical",
        height=360, barmode="stack",
    )
    st.plotly_chart(fig_w, use_container_width=True)
