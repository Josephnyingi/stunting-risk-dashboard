"""
dashboard.py  —  S2.T1.2 Stunting Risk Heatmap Dashboard
Run:  streamlit run dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data")
OUT_DIR  = Path("output")

st.set_page_config(
    page_title="Stunting Risk Dashboard · Rwanda",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data (cached) ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hh     = pd.read_csv(OUT_DIR / "households_scored.csv")
    sector = pd.read_csv(OUT_DIR / "sector_summary.csv")
    with open(DATA_DIR / "sectors.geojson") as f:
        sectors_geo = json.load(f)
    with open(OUT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return hh, sector, sectors_geo, metrics

hh, sector_summary, sectors_geo, metrics = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Stunting Risk")
    st.caption("S2.T1.2 · AIMS KTT Hackathon")
    st.divider()

    all_districts    = sorted(hh["district"].unique())
    sel_districts    = st.multiselect("District filter", all_districts, default=all_districts)
    risk_threshold   = st.slider("Risk threshold", 0.0, 1.0, 0.50, 0.05,
                                  help="Households at or above this score are flagged high-risk")
    show_points      = st.toggle("Show household points on map", value=True)

    st.divider()
    st.caption("Model performance (gold labels, n=300)")
    st.metric("AUC-ROC",   f"{metrics['auc_roc']:.3f}")
    st.metric("Precision", f"{metrics['precision']:.3f}")
    st.metric("Recall",    f"{metrics['recall']:.3f}")
    st.metric("F1",        f"{metrics['f1']:.3f}")

# ── Filter ───────────────────────────────────────────────────────────────────
filtered  = hh[hh["district"].isin(sel_districts)]
at_risk   = filtered[filtered["risk_score"] >= risk_threshold]
sec_filt  = sector_summary[sector_summary["district"].isin(sel_districts)]

# ── KPI row ──────────────────────────────────────────────────────────────────
st.title("Rwanda Childhood Stunting Risk Heatmap")
st.caption("Synthetic NISR-style data · 2,500 households · 5 districts · 15 sectors")

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
        # Sector choropleth
        fig_choro = px.choropleth_map(
            sec_filt,
            geojson=sectors_geo,
            locations="sector",
            featureidkey="properties.sector",
            color="pct_high_risk",
            color_continuous_scale="YlOrRd",
            range_color=(0, sec_filt["pct_high_risk"].max() + 0.05),
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

        # Overlay high-risk household scatter
        if show_points and len(at_risk) <= 600:
            fig_choro.add_trace(go.Scattermapbox(
                lat=at_risk["lat"],
                lon=at_risk["lon"],
                mode="markers",
                marker=dict(size=5, color="crimson", opacity=0.55),
                text=at_risk.apply(
                    lambda r: f"ID: {r['household_id']}<br>"
                              f"Score: {r['risk_score']:.2f}<br>"
                              f"Sector: {r['sector']}", axis=1
                ),
                hoverinfo="text",
                name="High-risk household",
            ))

        fig_choro.update_layout(margin={"r": 0, "l": 0, "t": 40, "b": 0},
                                coloraxis_colorbar_title="% High-risk")
        st.plotly_chart(fig_choro, use_container_width=True)

    with col_bar:
        # District bar
        dist_agg = (
            filtered.groupby("district")["risk_score"]
            .apply(lambda x: (x >= risk_threshold).mean())
            .reset_index()
            .rename(columns={"risk_score": "pct_high_risk"})
            .sort_values("pct_high_risk", ascending=True)
        )
        fig_bar = px.bar(
            dist_agg, x="pct_high_risk", y="district",
            orientation="h",
            color="pct_high_risk",
            color_continuous_scale="YlOrRd",
            labels={"pct_high_risk": "% High-risk", "district": ""},
            title="By district",
            height=540,
        )
        fig_bar.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin={"r": 10, "l": 0, "t": 40, "b": 0})
        st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════ TAB 2 : SECTOR ANALYSIS ════════════════════════════════════
with tab_sector:
    st.subheader("Sector-level risk summary")
    def _color_risk(val, vmin, vmax):
        t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
        r = int(255 * min(1, t * 2))
        g = int(255 * min(1, (1 - t) * 2))
        return f"background-color: rgb({r},{g},0); color: {'#fff' if t > 0.6 else '#000'}"

    _df = sec_filt.sort_values("pct_high_risk", ascending=False)
    _pmin, _pmax = _df["pct_high_risk"].min(), _df["pct_high_risk"].max()
    _smin, _smax = _df["avg_risk_score"].min(), _df["avg_risk_score"].max()
    st.dataframe(
        _df.style
            .map(lambda v: _color_risk(v, _pmin, _pmax), subset=["pct_high_risk"])
            .map(lambda v: _color_risk(v, _smin, _smax), subset=["avg_risk_score"]),
        use_container_width=True,
    )

    fig_scatter = px.scatter(
        sec_filt, x="avg_risk_score", y="pct_high_risk",
        size="n_households", color="district",
        text="sector",
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
        .style.map(lambda v: _color_risk(v, 0.0, 1.0), subset=["risk_score"]),
        use_container_width=True,
        height=480,
    )
    csv_bytes = at_risk[COLS].to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv_bytes,
                       "high_risk_households.csv", "text/csv")

# ═══════════════ TAB 4 : RISK DRIVERS ═══════════════════════════════════════
with tab_drivers:
    col_feat, col_hist = st.columns(2)

    with col_feat:
        feat_df = pd.DataFrame({
            "Feature": ["Water source", "Sanitation", "Income level",
                        "Meal frequency", "Children <5"],
            "Weight":  [0.30, 0.25, 0.25, 0.12, 0.08],
        }).sort_values("Weight")

        fig_feat = px.bar(
            feat_df, x="Weight", y="Feature",
            orientation="h",
            color="Weight",
            color_continuous_scale="RdYlGn_r",
            title="Feature weights (logistic regression coefficients)",
            height=350,
        )
        fig_feat.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_hist:
        fig_hist = px.histogram(
            filtered, x="risk_score", nbins=40,
            color="district",
            barmode="overlay",
            opacity=0.65,
            title="Risk score distribution by district",
            height=350,
        )
        fig_hist.add_vline(x=risk_threshold, line_dash="dash",
                           line_color="red",
                           annotation_text=f"Threshold {risk_threshold:.2f}")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Water source breakdown
    st.subheader("Water source mix in high-risk households")
    water_counts = (
        at_risk.groupby(["water_source", "district"])
        .size().reset_index(name="count")
    )
    fig_water = px.bar(water_counts, x="water_source", y="count",
                       color="district", barmode="group",
                       title="High-risk households by water source and district",
                       height=320)
    st.plotly_chart(fig_water, use_container_width=True)
