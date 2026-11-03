#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 World map of scientific production by country, with an embedded Top-10 table
=============================================================================

Input : bibliometric database (Excel, one row per document) containing a
        country field (corresponding author / country of affiliation) and,
        optionally, a topic field produced by a topic model (LDA / BERTopic).

Output: publication-ready choropleth map (no title, English labels) with the
        Top-10 most productive countries reported in a table placed inside
        the map frame. PNG + TIFF + PDF at high resolution.

Behaviour
---------
* The script looks for the topic column (see TOPIC_COLUMN / TOPIC_CANDIDATES).
    - If it is found, the map can be restricted to one topic (TOPIC_FILTER)
      and/or one map per topic can be exported (MAPS_BY_TOPIC = True).
    - If it is NOT found, the script simply maps the whole corpus and prints
      a warning. No error is raised.
* Country names are converted to ISO alpha-3 codes with `pycountry`, plus a
  manual dictionary for the usual Scopus / Web of Science spellings.

Requirements
------------
    pip install pandas openpyxl geopandas matplotlib pycountry

Author: J. De La Hoz Maestre
=============================================================================
"""

from __future__ import annotations

import os
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize

import geopandas as gpd
import pycountry

# =============================================================================
# 1. USER SETTINGS
# =============================================================================

EXCEL_PATH = "BD.xlsx"          # input database
SHEET_NAME = 0                  # 0 = first sheet, or "Sheet 1"

COUNTRY_COLUMN = "Country"      # column with the country of the document

# --- topic column -----------------------------------------------------------
TOPIC_COLUMN = "Topic"          # preferred name of the topic column
TOPIC_CANDIDATES = [            # alternative names searched automatically
    "Topic", "topic", "Topic_ID", "Topic_Label", "Topic_Name",
    "Tópico", "Topico", "Cluster", "cluster", "LDA_Topic", "BERTopic",
]
TOPIC_FILTER = None             # None = all documents; e.g. 3 or "Topic 3"
MAPS_BY_TOPIC = False           # True = one additional map per topic
DROP_OUTLIER_TOPIC = True       # drop BERTopic outliers (topic == -1)

# --- figure -----------------------------------------------------------------
OUTPUT_DIR = "figures"
OUTPUT_STEM = "world_map_production"
TOP_N = 10                    # rows of the embedded table
FIG_SIZE = (14, 7.4)            # inches
DPI = 600
SAVE_FORMATS = ("png", "tiff", "pdf")

COLOR_SCALE = "log"             # "log" | "linear" | "classes"
CLASS_BREAKS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]  # for "classes"
CMAP_NAME = "YlOrRd"            # sequential colormap
NO_DATA_COLOR = "#EFEFEF"       # countries without publications
BORDER_COLOR = "#666666"
PROJECTION = "+proj=robin"      # Robinson; use None to keep WGS84
DROP_ANTARCTICA = True

CBAR_LABEL = "Number of publications"
TABLE_TITLE = "Top 10 most productive countries"   # set to None to remove
TABLE_POSITION = (0.055, 0.085, 0.285, 0.340)      # [left, bottom, w, h]

# Which table columns are filled with the map colour scale.
#   None      -> the whole row (0 = #, 1 = Country, 2 = Records, 3 = %)
#   (2,)      -> only the "Records" column
#   (0, 2)    -> rank and records
TABLE_COLOR_COLUMNS = None

# --- world geometry (Natural Earth 1:110m) ----------------------------------
GEOJSON_LOCAL = "ne_110m_admin_0_countries.geojson"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)

# Scopus / WoS spellings that pycountry may not resolve directly
MANUAL_ISO3 = {
    "iran": "IRN",
    "south korea": "KOR",
    "north korea": "PRK",
    "russia": "RUS",
    "vietnam": "VNM",
    "turkey": "TUR",
    "czech republic": "CZE",
    "slovak republic": "SVK",
    "hong kong": "HKG",
    "macao": "MAC",
    "macau": "MAC",
    "taiwan": "TWN",
    "usa": "USA",
    "u.s.a.": "USA",
    "united states of america": "USA",
    "uk": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "northern ireland": "GBR",
    "bolivia": "BOL",
    "venezuela": "VEN",
    "tanzania": "TZA",
    "syria": "SYR",
    "laos": "LAO",
    "brunei": "BRN",
    "moldova": "MDA",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "cape verde": "CPV",
    "swaziland": "SWZ",
    "burma": "MMR",
    "palestine": "PSE",
    "kosovo": "XKX",
    "democratic republic congo": "COD",
    "peoples r china": "CHN",
    "china mainland": "CHN",
}


# =============================================================================
# 2. HELPERS
# =============================================================================
def to_iso3(name: str) -> str | None:
    """Convert a country name into an ISO alpha-3 code."""
    if not isinstance(name, str):
        return None
    raw = name.strip()
    if not raw or raw.upper() in {"NA", "N/A", "NONE", "NOTREPORTED", "-"}:
        return None

    key = raw.lower().replace(".", "").strip()
    if key in MANUAL_ISO3:
        return MANUAL_ISO3[key]

    try:
        return pycountry.countries.lookup(raw).alpha_3
    except LookupError:
        pass

    # fall back to the part before a comma: "Iran, Islamic Republic of" -> "Iran"
    head = raw.split(",")[0].strip()
    if head.lower() in MANUAL_ISO3:
        return MANUAL_ISO3[head.lower()]
    try:
        return pycountry.countries.lookup(head).alpha_3
    except LookupError:
        return None


def iso3_to_name(code: str) -> str:
    """Short, readable English country name for the table."""
    pretty = {
        "IRN": "Iran", "KOR": "South Korea", "PRK": "North Korea",
        "RUS": "Russia", "VNM": "Vietnam", "USA": "United States",
        "GBR": "United Kingdom", "TWN": "Taiwan", "HKG": "Hong Kong",
        "CZE": "Czech Republic", "ARE": "United Arab Emirates",
        "TUR": "Turkey", "SAU": "Saudi Arabia", "NZL": "New Zealand",
        "ZAF": "South Africa",
    }
    if code in pretty:
        return pretty[code]
    try:
        country = pycountry.countries.get(alpha_3=code)
        return country.common_name if hasattr(country, "common_name") else country.name
    except Exception:
        return code


def resolve_topic_column(df: pd.DataFrame) -> str | None:
    """Return the name of the topic column, or None if the corpus has none."""
    if TOPIC_COLUMN in df.columns:
        return TOPIC_COLUMN
    for cand in TOPIC_CANDIDATES:
        if cand in df.columns:
            return cand
    lower = {c.lower(): c for c in df.columns}
    for cand in TOPIC_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_world() -> gpd.GeoDataFrame:
    """Load the Natural Earth country layer (local copy or download once)."""
    if not os.path.exists(GEOJSON_LOCAL):
        print(f"[info] downloading world geometry -> {GEOJSON_LOCAL}")
        urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_LOCAL)

    world = gpd.read_file(GEOJSON_LOCAL)
    iso_col = "ISO_A3_EH" if "ISO_A3_EH" in world.columns else "ISO_A3"
    world = world.rename(columns={iso_col: "iso3"})

    # Natural Earth stores -99 for a few disputed / dependent territories
    fix = {"France": "FRA", "Norway": "NOR", "Kosovo": "XKX",
           "Somaliland": "SOM", "N. Cyprus": "CYP"}
    world.loc[world["iso3"].isin(["-99", "-099"]), "iso3"] = (
        world.loc[world["iso3"].isin(["-99", "-099"]), "NAME"].map(fix)
    )

    if DROP_ANTARCTICA:
        world = world[world["iso3"] != "ATA"]
    return world[["iso3", "NAME", "geometry"]]


def count_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Documents per ISO-3 country code."""
    iso = df[COUNTRY_COLUMN].map(to_iso3)

    unmatched = sorted(set(df.loc[iso.isna(), COUNTRY_COLUMN].dropna().astype(str)))
    if unmatched:
        print(f"[warn] {len(unmatched)} country label(s) not matched: {unmatched}")

    counts = (
        iso.dropna()
        .value_counts()
        .rename_axis("iso3")
        .reset_index(name="records")
    )
    total = counts["records"].sum()
    counts["share"] = 100.0 * counts["records"] / total
    counts["country"] = counts["iso3"].map(iso3_to_name)
    return counts.sort_values("records", ascending=False).reset_index(drop=True)


# =============================================================================
# 3. FIGURE
# =============================================================================
def draw_map(counts: pd.DataFrame, world: gpd.GeoDataFrame, stem: str) -> None:
    """Choropleth map with the Top-N table inside the map frame. No title."""
    gdf = world.merge(counts[["iso3", "records"]], on="iso3", how="left")
    if PROJECTION:
        gdf = gdf.to_crs(PROJECTION)

    vmax = float(counts["records"].max())
    cmap = mpl.colormaps[CMAP_NAME]

    if COLOR_SCALE == "log":
        norm = LogNorm(vmin=1, vmax=vmax)
    elif COLOR_SCALE == "classes":
        breaks = [b for b in CLASS_BREAKS if b <= vmax] + [vmax]
        norm = BoundaryNorm(sorted(set(breaks)), ncolors=cmap.N, clip=True)
    else:
        norm = Normalize(vmin=1, vmax=vmax)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # countries without records
    gdf[gdf["records"].isna()].plot(
        ax=ax, color=NO_DATA_COLOR, edgecolor=BORDER_COLOR, linewidth=0.25
    )
    # countries with records
    gdf[gdf["records"].notna()].plot(
        ax=ax, column="records", cmap=cmap, norm=norm,
        edgecolor=BORDER_COLOR, linewidth=0.25,
    )

    ax.set_axis_off()
    ax.margins(0)

    # ---- colour bar --------------------------------------------------------
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        fraction=0.035, pad=0.02, shrink=0.42, aspect=26, extend="neither",
    )
    cbar.set_label(CBAR_LABEL, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    if COLOR_SCALE == "log":
        ticks = [t for t in (1, 5, 10, 50, 100, 500, 1000, 5000) if t <= vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:,}" for t in ticks])
    cbar.outline.set_linewidth(0.4)

    # ---- Top-N table inside the map ---------------------------------------
    top = counts.head(TOP_N).copy()
    cells = [
        [str(i + 1), r.country, f"{int(r.records):,}", f"{r.share:.1f}"]
        for i, r in enumerate(top.itertuples(index=False))
    ]

    tax = fig.add_axes(TABLE_POSITION)
    tax.set_axis_off()
    tbl = tax.table(
        cellText=cells,
        colLabels=["#", "Country", "Records", "%"],
        colWidths=[0.12, 0.50, 0.21, 0.17],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.22)

    # the table shares the colour scale of the map: each data row is filled
    # with cmap(norm(records)), so a cell and its country match in colour
    values = top["records"].tolist()

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#B0B0B0")
        cell.set_linewidth(0.4)
        if row == 0:                                   # header
            cell.set_facecolor("#4A4A4A")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            rgba = cmap(norm(values[row - 1]))
            if TABLE_COLOR_COLUMNS is None:
                paint = True
            else:
                paint = col in TABLE_COLOR_COLUMNS
            if paint:
                cell.set_facecolor(rgba)
                # luminance-based text colour keeps the numbers readable
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                cell.set_text_props(color="white" if lum < 0.45 else "black")
            else:
                cell.set_facecolor("#FFFFFF")
        if col == 1:
            cell.set_text_props(ha="left")
            cell._text.set_x(0.04)

    if TABLE_TITLE:
        tax.set_title(TABLE_TITLE, fontsize=9.5, fontweight="bold", pad=6)

    # ---- export ------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in SAVE_FORMATS:
        path = os.path.join(OUTPUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", pad_inches=0.12)
        print(f"[ok] {path}")
    plt.close(fig)


# =============================================================================
# 4. MAIN
# =============================================================================
def main() -> None:
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    print(f"[info] {len(df):,} documents x {df.shape[1]} fields")

    if COUNTRY_COLUMN not in df.columns:
        raise KeyError(f"Column '{COUNTRY_COLUMN}' not found in {EXCEL_PATH}")

    # --- topic column ------------------------------------------------------
    topic_col = resolve_topic_column(df)
    if topic_col is None:
        print("[warn] no topic column found -> mapping the whole corpus")
    else:
        print(f"[info] topic column: '{topic_col}' "
              f"({df[topic_col].nunique()} topics)")
        if DROP_OUTLIER_TOPIC:
            before = len(df)
            df = df[~df[topic_col].astype(str).str.strip().isin(["-1", "-1.0"])]
            if len(df) < before:
                print(f"[info] {before - len(df):,} outlier document(s) removed")
        if TOPIC_FILTER is not None:
            wanted = TOPIC_FILTER if isinstance(TOPIC_FILTER, (list, tuple, set)) \
                else [TOPIC_FILTER]
            df = df[df[topic_col].isin(wanted)]
            print(f"[info] topic filter {list(wanted)} -> {len(df):,} documents")

    world = load_world()

    # --- global map --------------------------------------------------------
    counts = count_by_country(df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts.to_csv(os.path.join(OUTPUT_DIR, "country_counts.csv"), index=False)

    print(f"[info] {len(counts)} countries; "
          f"leader: {counts.iloc[0].country} ({int(counts.iloc[0].records):,})")
    draw_map(counts, world, OUTPUT_STEM)

    # --- one map per topic (optional) --------------------------------------
    if MAPS_BY_TOPIC and topic_col is not None:
        for value in sorted(df[topic_col].dropna().unique(), key=str):
            sub = df[df[topic_col] == value]
            if sub.empty:
                continue
            slug = str(value).replace(" ", "_").replace("/", "-")[:40]
            print(f"\n[info] topic {value}: {len(sub):,} documents")
            draw_map(count_by_country(sub), world, f"{OUTPUT_STEM}_topic_{slug}")


if __name__ == "__main__":
    main()
