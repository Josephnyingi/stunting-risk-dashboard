"""
generate_data.py
Reproducible synthetic data generator for S2.T1.2 Stunting Risk Heatmap.
Seed: 42.  Runtime: <2 min on laptop.
Usage: python generate_data.py
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

RNG = np.random.default_rng(42)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── District / Sector definitions with explicit bounding boxes ───────────────
# bbox = (lon_min, lat_min, lon_max, lat_max)
DISTRICTS = {
    "Nyarugenge": {
        "center": (-1.950, 30.060),
        "bbox":   (30.020, -1.990, 30.100, -1.910),
        "sectors": {
            "Gitega":     (30.020, -1.990, 30.060, -1.950),
            "Kigali":     (30.060, -1.990, 30.100, -1.950),
            "Nyarugenge": (30.020, -1.950, 30.100, -1.910),
        },
        "urban_weight": 0.90,
        "stunting_base": 0.18,   # Kigali City ~18-22% (NISR DHS 2019-20)
    },
    "Gasabo": {
        "center": (-1.875, 30.112),
        "bbox":   (30.057, -1.930, 30.167, -1.820),
        "sectors": {
            "Kimironko": (30.057, -1.930, 30.112, -1.875),
            "Remera":    (30.112, -1.930, 30.167, -1.875),
            "Kacyiru":   (30.057, -1.875, 30.167, -1.820),
        },
        "urban_weight": 0.80,
        "stunting_base": 0.21,
    },
    "Kicukiro": {
        "center": (-2.005, 30.103),
        "bbox":   (30.058, -2.050, 30.148, -1.960),
        "sectors": {
            "Niboye":  (30.058, -2.050, 30.103, -2.005),
            "Gahanga": (30.103, -2.050, 30.148, -2.005),
            "Masaka":  (30.058, -2.005, 30.148, -1.960),
        },
        "urban_weight": 0.72,
        "stunting_base": 0.24,
    },
    "Nyanza": {
        "center": (-2.352, 29.748),
        "bbox":   (29.608, -2.492, 29.888, -2.212),
        "sectors": {
            "Busasamana": (29.608, -2.492, 29.748, -2.352),
            "Cyabakamyi": (29.748, -2.492, 29.888, -2.352),
            "Kibilizi":   (29.608, -2.352, 29.888, -2.212),
        },
        "urban_weight": 0.18,
        "stunting_base": 0.38,   # Southern Province ~38% (NISR DHS 2019-20)
    },
    "Musanze": {
        "center": (-1.498, 29.633),
        "bbox":   (29.513, -1.618, 29.753, -1.378),
        "sectors": {
            "Muhoza":   (29.513, -1.618, 29.633, -1.498),
            "Kinigi":   (29.633, -1.618, 29.753, -1.498),
            "Shingiro": (29.513, -1.498, 29.753, -1.378),
        },
        "urban_weight": 0.28,
        "stunting_base": 0.41,   # Northern Province ~40% (NISR DHS 2019-20)
    },
}

WATER_SOURCES   = ["piped", "protected_well", "unprotected_well", "river_lake"]
SANITATION_TIERS = ["improved", "basic", "limited", "open_defecation"]
INCOME_BANDS    = ["high", "medium", "low", "very_low"]

WATER_RISK  = {"piped": 0.0, "protected_well": 0.33, "unprotected_well": 0.67, "river_lake": 1.0}
SANIT_RISK  = {"improved": 0.0, "basic": 0.33, "limited": 0.67, "open_defecation": 1.0}
INCOME_RISK = {"high": 0.0, "medium": 0.33, "low": 0.67, "very_low": 1.0}

N_HOUSEHOLDS = 2500
N_GOLD = 300


def bbox_to_polygon(lon_min, lat_min, lon_max, lat_max):
    return [[lon_min, lat_min], [lon_max, lat_min],
            [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]


def build_geojson():
    district_features = []
    sector_features = []

    for district, info in DISTRICTS.items():
        d_bbox = info["bbox"]
        district_features.append({
            "type": "Feature",
            "properties": {"district": district, "stunting_base": info["stunting_base"]},
            "geometry": {"type": "Polygon",
                         "coordinates": [bbox_to_polygon(*d_bbox)]},
        })
        for sector, s_bbox in info["sectors"].items():
            sector_features.append({
                "type": "Feature",
                "properties": {"district": district, "sector": sector},
                "geometry": {"type": "Polygon",
                             "coordinates": [bbox_to_polygon(*s_bbox)]},
            })

    with open(DATA_DIR / "districts.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": district_features}, f)
    with open(DATA_DIR / "sectors.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": sector_features}, f)
    print("GeoJSON written.")


def stunting_prob(row: dict) -> float:
    w = (0.35 * WATER_RISK[row["water_source"]]
         + 0.30 * SANIT_RISK[row["sanitation_tier"]]
         + 0.25 * INCOME_RISK[row["income_band"]]
         + 0.15 * (1.0 - (float(row["avg_meal_count"]) - 1.0) / 4.0)
         + 0.10 * min(int(row["children_under5"]) / 5.0, 1.0))
    base = DISTRICTS[row["district"]]["stunting_base"]
    return float(np.clip(0.4 * base + 0.6 * w, 0.0, 1.0))


def build_households():
    district_sizes = {
        "Nyarugenge": 600, "Gasabo": 650, "Kicukiro": 550,
        "Nyanza": 400, "Musanze": 300,
    }
    rows = []
    hh_id = 1

    for district, n in district_sizes.items():
        info = DISTRICTS[district]
        urban_w = info["urban_weight"]
        sectors = list(info["sectors"].keys())

        for _ in range(n):
            sector = sectors[RNG.integers(len(sectors))]
            lon_min, lat_min, lon_max, lat_max = info["sectors"][sector]
            is_urban = RNG.random() < urban_w

            lat = float(RNG.uniform(lat_min, lat_max))
            lon = float(RNG.uniform(lon_min, lon_max))

            if is_urban:
                water = RNG.choice(WATER_SOURCES, p=[0.55, 0.25, 0.15, 0.05])
                sanit = RNG.choice(SANITATION_TIERS, p=[0.55, 0.25, 0.15, 0.05])
                income = RNG.choice(INCOME_BANDS, p=[0.20, 0.40, 0.30, 0.10])
                meals = float(RNG.integers(2, 6))
                children = int(RNG.integers(0, 4))
            else:
                water = RNG.choice(WATER_SOURCES, p=[0.10, 0.25, 0.35, 0.30])
                sanit = RNG.choice(SANITATION_TIERS, p=[0.10, 0.25, 0.35, 0.30])
                income = RNG.choice(INCOME_BANDS, p=[0.05, 0.20, 0.45, 0.30])
                meals = float(RNG.integers(1, 4))
                children = int(RNG.integers(0, 6))

            row = {
                "household_id": f"HH{hh_id:05d}",
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "district": district,
                "sector": sector,
                "children_under5": children,
                "avg_meal_count": meals,
                "water_source": water,
                "sanitation_tier": sanit,
                "income_band": income,
                "urban_rural": "urban" if is_urban else "rural",
            }
            row["_prob"] = stunting_prob(row)
            rows.append(row)
            hh_id += 1

    df = pd.DataFrame(rows)
    df.drop(columns=["_prob"]).to_csv(DATA_DIR / "households.csv", index=False)

    # Gold labels: 150 positive (high prob) + 150 negative (low prob)
    pos = df[df["_prob"] >= 0.45].sample(150, random_state=42)
    neg = df[df["_prob"] <= 0.25].sample(150, random_state=42)
    gold = pd.concat([pos[["household_id"]], neg[["household_id"]]]).copy()
    gold["stunting_flag"] = [1] * 150 + [0] * 150
    gold.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(
        DATA_DIR / "gold_stunting_flag.csv", index=False
    )

    print(f"households.csv        : {len(df)} rows")
    print(f"gold_stunting_flag.csv: {len(gold)} rows  "
          f"(prevalence = {gold['stunting_flag'].mean():.0%})")
    return df


if __name__ == "__main__":
    build_geojson()
    build_households()
    print(f"\nAll files written to {DATA_DIR}/")
