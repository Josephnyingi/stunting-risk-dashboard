"""
export_printable.py
Generate printable A4 'Umudugudu pages' — one PDF per sector.
Each page: header, summary stats, top-10 anonymised high-risk households,
top-3 drivers, intervention hint, bilingual labels (English / Kinyarwanda),
and annotation/escalation block for paper workflow.

Run: python export_printable.py
"""
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT_DIR = Path("output")
PDF_DIR = Path("printable")
PDF_DIR.mkdir(exist_ok=True)

# ── Kinyarwanda labels ───────────────────────────────────────────────────────
KIN = {
    "header_org":   "MINISANTE / NISR",
    "header_title": "Raporo y'Ibyago byo Kugwingira · Umudugudu",
    "monthly":      "Raporo ya Buri Kwezi",
    "district":     "Akarere",
    "sector":       "Umurenge",
    "total_hh":     "Inzu zose",
    "high_risk":    "Ibyago bikabije",
    "avg_score":    "Amanota asanzwe",
    "pct_high":     "% Ibyago",
    "rank":         "N°",
    "anon_id":      "Nimero y'Inzu",
    "score":        "Amanota",
    "children":     "Abana <5y",
    "drivers":      "Impamvu 3",
    "intervention": "Ubufasha",
    "privacy":      (
        "IBANGA · Iri rupapuro ni ibanga. "
        "Ntirugomba gusangirwa aho abantu benshi bari."
    ),
    "printed_by":   "Yasinye:",
    "reviewed_by":  "Yasuzumye:",
    "escalated":    "Byoherejwe MINISANTE: □ Yego  □ Oya    Impamvu:",
    "annotation":   "Ibisobanuro:",
}

RISK_COLORS = {
    "critical": colors.HexColor("#b22222"),
    "high":     colors.HexColor("#e05c00"),
    "moderate": colors.HexColor("#e6a817"),
    "low":      colors.HexColor("#4a7c59"),
}
HEADER_RED = colors.HexColor("#8b0000")


def tier(score: float) -> str:
    if score >= 0.75: return "critical"
    if score >= 0.55: return "high"
    if score >= 0.35: return "moderate"
    return "low"


def generate_sector_pdf(sector_name: str, district_name: str, df_sector: pd.DataFrame):
    top10     = df_sector.nlargest(10, "risk_score").reset_index(drop=True)
    pdf_path  = PDF_DIR / f"sector_{sector_name.replace(' ', '_')}.pdf"
    today_str = date.today().strftime("%d %B %Y")

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.4*cm,  bottomMargin=1.4*cm,
    )

    # ── Styles ───────────────────────────────────────────────────────────────
    base   = getSampleStyleSheet()
    def style(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    S = {
        "org":     style("org",  fontSize=8,  fontName="Helvetica",
                         alignment=TA_CENTER, textColor=colors.HexColor("#555")),
        "title":   style("tit",  fontSize=13, fontName="Helvetica-Bold",
                         alignment=TA_CENTER, spaceAfter=2),
        "sub":     style("sub",  fontSize=9,  fontName="Helvetica",
                         alignment=TA_CENTER, spaceAfter=2),
        "label":   style("lbl",  fontSize=8,  fontName="Helvetica-Bold"),
        "small":   style("sm",   fontSize=7,  fontName="Helvetica",
                         textColor=colors.HexColor("#555")),
        "privacy": style("prv",  fontSize=8,  fontName="Helvetica-Bold",
                         textColor=colors.HexColor("#cc0000"),
                         alignment=TA_CENTER),
    }

    story = []

    # ── HEADER ───────────────────────────────────────────────────────────────
    story.append(Paragraph(KIN["header_org"], S["org"]))
    story.append(Paragraph(
        f"{KIN['monthly']} — {date.today().strftime('%B %Y')} &nbsp;|&nbsp; {today_str}",
        S["org"]
    ))
    story.append(HRFlowable(width="100%", thickness=2.5, color=HEADER_RED))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        f"{KIN['district']}: <b>{district_name}</b> &nbsp;·&nbsp; "
        f"{KIN['sector']}: <b>{sector_name}</b>",
        S["title"]
    ))
    story.append(Paragraph(KIN["header_title"], S["sub"]))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("⚠ " + KIN["privacy"], S["privacy"]))
    story.append(Spacer(1, 0.25*cm))

    # ── SUMMARY STATS ────────────────────────────────────────────────────────
    n_total = len(df_sector)
    n_high  = int((df_sector["risk_score"] >= 0.50).sum())
    pct_h   = n_high / max(n_total, 1)
    avg_s   = df_sector["risk_score"].mean()

    stats_data = [
        [f"{KIN['total_hh']} / Total HH", str(n_total),
         f"{KIN['high_risk']} / High-risk", str(n_high)],
        [f"{KIN['avg_score']} / Avg score", f"{avg_s:.2f}",
         f"{KIN['pct_high']} / % high",    f"{pct_h:.0%}"],
    ]
    stats_tbl = Table(stats_data, colWidths=[5.5*cm, 2.5*cm, 5.5*cm, 2.5*cm])
    stats_tbl.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1),  "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("PADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(stats_tbl)
    story.append(Spacer(1, 0.30*cm))

    # ── TOP-10 TABLE ─────────────────────────────────────────────────────────
    story.append(Paragraph(
        "<b>Top 10 Households · Highest Stunting Risk</b> "
        "(anonymisé / ibyoherezwa nimero gusa)",
        S["label"]
    ))
    story.append(Spacer(1, 0.15*cm))

    header_row = [
        KIN["rank"], KIN["anon_id"],
        f"{KIN['score']}\n(0–1)",
        KIN["children"],
        "Top-3 Drivers\n(Impamvu 3)",
        "Intervention\n(Ubufasha)",
    ]
    col_w = [0.8*cm, 1.8*cm, 1.8*cm, 1.3*cm, 6.2*cm, 5.4*cm]
    table_data = [header_row]

    row_score_colors = []
    for i, row in top10.iterrows():
        anon  = f"{sector_name[:3].upper()}-{i+1:02d}"
        drvrs = row.get("top_drivers", "N/A")
        # newline-separated for readability
        drvrs_fmt = drvrs.replace(" | ", "\n") if isinstance(drvrs, str) else "N/A"
        interv = row.get("intervention", "—")
        score  = float(row["risk_score"])
        table_data.append([
            str(i + 1), anon, f"{score:.3f}",
            str(int(row["children_under5"])),
            drvrs_fmt, interv,
        ])
        row_score_colors.append(RISK_COLORS[tier(score)])

    tbl = Table(table_data, colWidths=col_w, repeatRows=1)
    tbl_style = [
        ("FONTSIZE",     (0, 0), (-1, -1), 7),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("BACKGROUND",   (0, 0), (-1,  0), HEADER_RED),
        ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
        ("ALIGN",        (0, 0), (3, -1),  "CENTER"),
        ("ALIGN",        (4, 0), (5, -1),  "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#fff7f7")]),
        ("PADDING",      (0, 0), (-1, -1), 3),
    ]
    for i, rc in enumerate(row_score_colors, start=1):
        tbl_style.extend([
            ("BACKGROUND", (2, i), (2, i), rc),
            ("TEXTCOLOR",  (2, i), (2, i), colors.white),
        ])
    tbl.setStyle(TableStyle(tbl_style))
    story.append(tbl)

    # ── ANNOTATION / ESCALATION BLOCK ────────────────────────────────────────
    story.append(Spacer(1, 0.45*cm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.lightgrey))
    story.append(Spacer(1, 0.10*cm))
    for line in [
        f"{KIN['printed_by']} ________________  |  {KIN['reviewed_by']} ________________  "
        f"|  Itariki / Date: ________________",
        f"{KIN['annotation']} "
        "______________________________________________________________________",
        f"{KIN['escalated']} "
        "___________________________________________",
    ]:
        story.append(Paragraph(line, S["small"]))
        story.append(Spacer(1, 0.12*cm))

    # ── LEGEND ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.05*cm))
    story.append(Paragraph(
        "<font color='#b22222'>■ Critical ≥0.75</font>  "
        "<font color='#e05c00'>■ High 0.55–0.74</font>  "
        "<font color='#e6a817'>■ Moderate 0.35–0.54</font>  "
        "<font color='#4a7c59'>■ Low &lt;0.35</font>",
        S["small"]
    ))

    doc.build(story)
    print(f"  {pdf_path.name}")


if __name__ == "__main__":
    print("Loading scored households …")
    hh = pd.read_csv(OUT_DIR / "households_scored.csv")

    print("Generating A4 PDFs …")
    for (district, sector), group in hh.groupby(["district", "sector"]):
        generate_sector_pdf(sector, district, group)

    pdfs = list(PDF_DIR.glob("*.pdf"))
    print(f"\n{len(pdfs)} PDFs written to {PDF_DIR}/")
