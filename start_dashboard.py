# app.py
# -----------------------------------------------------------------------------
# Plotly Dash Grundgerüst – Styling via assets/styles.css
# - CSV: section_sample.csv (Zeitreihen + ggf. Wetterfeatures)
# - CSV: summary.csv (Mapping load_id -> Sector_group)
# - Dropdown: Branche (Sector_group)
# - Dropdown: Target-Load (gefiltert nach Branche)
# - Auswahlmenü einklappbar (Collapse mit Pfeil)
# - Plotly: Zeitreihe
# - Plotly: Self-Similarity via analysis/self_similarity/run_self_similarity.py -> (fig, interpretation)
# - Plotly: Min/Max/Median via analysis/min_max_median.py  -> (fig, capture)
# - Plotly: Weather Dependence via analysis/weather_dependence.py -> (fig, capture)
# - Unter Self-Similarity / MinMax / Weather: Textfeld mit Interpretation/Capture
# - Refresh Badge rechts in der Navbar (Client-side Reload)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
import plotly.graph_objects as go

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

# =============================================================================
# Optional: Projekt-Root in sys.path, damit "analysis.*" sicher importierbar ist
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# External analysis calls (keine Logik-Duplikation im Dashboard)
from analysis.self_similarity.run_self_similarity import generate_self_similarity_plot  # noqa: E402
from analysis.min_max_median import generate_distribution_comparison  # noqa: E402

# Weather dependence (ausgelagert)
from analysis.weather_dependence import (  # noqa: E402
    generate_distribution_comparison as generate_weather_dependence_plot,
)

# =============================================================================
# Logging
# =============================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("timeseries_dash")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class AppConfig:
    title: str = "EnerBench | Time Series Analysis Dashboard"
    theme = dbc.themes.CYBORG
    host: str = os.getenv("DASH_HOST", "127.0.0.1")
    port: int = int(os.getenv("DASH_PORT", "8050"))
    debug: bool = os.getenv("DASH_DEBUG", "0").lower() in ("1", "true", "yes")
    fig_height: int = 520


CFG = AppConfig()


# =============================================================================
# Data loading
# =============================================================================
DEFAULT_DATA_FILE = Path(r"D:\EnerBench\RawData\section_sample.csv")
DEFAULT_SUMMARY_FILE = Path(r"D:\EnerBench\RawData\summary.csv")


def resolve_data_file() -> Path:
    env = os.getenv("DATA_FILE")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"DATA_FILE existiert nicht: {p}")
        return p

    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE

    raise FileNotFoundError(
        "Keine Zeitreihen-CSV gefunden.\n"
        "Lösung:\n"
        "  1) setze ENV DATA_FILE auf den Pfad zur CSV\n"
        f"Default geprüft: {DEFAULT_DATA_FILE}"
    )


def resolve_summary_file() -> Path:
    env = os.getenv("SUMMARY_FILE")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"SUMMARY_FILE existiert nicht: {p}")
        return p

    if DEFAULT_SUMMARY_FILE.exists():
        return DEFAULT_SUMMARY_FILE

    raise FileNotFoundError(
        "Keine summary.csv gefunden.\n"
        "Lösung:\n"
        "  1) setze ENV SUMMARY_FILE auf den Pfad zur summary.csv\n"
        f"Default geprüft: {DEFAULT_SUMMARY_FILE}"
    )


def load_timeseries_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Lade Zeitreihen-CSV: %s", path)
    df = pd.read_csv(path, low_memory=False)

    time_col = df.columns[0]
    df = df.rename(columns={time_col: "timestamp"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    before = len(df)
    df = df[df["timestamp"].notna()].copy()
    removed = before - len(df)
    if removed:
        LOGGER.info("Rows ohne gültigen Timestamp entfernt: %s", removed)

    df = df.sort_values("timestamp")
    return df


def load_summary_csv(path: Path) -> pd.DataFrame:
    """
    Erwartet mindestens:
      - eine Spalte für Load-ID (z.B. 'load_id', 'id', 'Load_ID', ...)
      - eine Spalte 'Sector_group' (case-insensitive, ggf. 'sector_group')
    """
    LOGGER.info("Lade Summary CSV: %s", path)
    s = pd.read_csv(path, low_memory=False)

    cols_lower = {c.lower(): c for c in s.columns}

    sector_col = None
    for key in ("sector_group", "sector group", "sector"):
        if key in cols_lower:
            sector_col = cols_lower[key]
            break
    if sector_col is None:
        raise KeyError(f"summary.csv: Keine Spalte 'Sector_group' gefunden. Vorhanden: {list(s.columns)}")

    id_col = None
    for key in ("load_id", "loadid", "id", "load", "loadid ", "load id"):
        if key in cols_lower:
            id_col = cols_lower[key]
            break
    if id_col is None:
        for c in s.columns:
            if "id" in c.lower():
                id_col = c
                break
    if id_col is None:
        raise KeyError(f"summary.csv: Keine ID-Spalte gefunden. Vorhanden: {list(s.columns)}")

    out = s[[id_col, sector_col]].copy()
    out = out.rename(columns={id_col: "load_id", sector_col: "Sector_group"})

    out["load_id"] = out["load_id"].astype(str).str.strip()
    out["Sector_group"] = out["Sector_group"].astype(str).str.strip()

    out = out[(out["load_id"] != "") & (out["Sector_group"] != "")]
    out = out.drop_duplicates(subset=["load_id"], keep="first")
    return out


def find_load_columns(df: pd.DataFrame) -> List[str]:
    load_cols = [c for c in df.columns if c.startswith("load_")]
    if not load_cols:
        raise KeyError("Keine Spalten gefunden, die mit 'load_' beginnen.")
    return sorted(load_cols)


def label_for_load(col: str) -> str:
    return col.replace("load_", "Load ")


def load_id(col: str) -> str:
    return col.replace("load_", "")


# =============================================================================
# Figure styling
# =============================================================================
BG = "#0b0f14"
GRID = "rgba(255,255,255,0.10)"
AXIS = "rgba(255,255,255,0.25)"
TEXT = "#ffffff"


def graph_config() -> dict:
    return {
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    }


def style_figure(fig: go.Figure, title: str, x_label: str, y_label: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, Segoe UI, Arial", size=12, color=TEXT),
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", font=dict(color=TEXT)),
        autosize=True,
        height=CFG.fig_height,
        margin=dict(l=60, r=20, t=60, b=55),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.10,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        ),
    )
    fig.update_xaxes(title_text=x_label, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT))
    fig.update_yaxes(title_text=y_label, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT))
    return fig


def add_frame(fig: go.Figure, width: float = 1.5) -> go.Figure:
    frame = dict(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color=AXIS, width=width),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    shapes.append(frame)
    fig.update_layout(shapes=shapes)
    return fig


def apply_dashboard_theme(
    fig: go.Figure,
    *,
    height: int = CFG.fig_height,
    showlegend: Optional[bool] = None,
) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, Segoe UI, Arial", size=12, color=TEXT),
        height=height,
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)

    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT))
    return fig


def fig_load_timeseries(df: pd.DataFrame, load_col: str) -> go.Figure:
    dff = df[["timestamp", load_col]].copy()
    dff[load_col] = pd.to_numeric(dff[load_col], errors="coerce")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dff["timestamp"],
            y=dff[load_col],
            mode="lines",
            name=label_for_load(load_col),
            line=dict(width=2),
        )
    )
    fig = style_figure(fig, "Zeitreihe", "Zeit", "Leistung (kW)")
    fig.update_layout(showlegend=False)
    return add_frame(fig)


# =============================================================================
# UI helpers
# =============================================================================
def panel(title: str, info_id: Optional[str] = None, info_text: Optional[str] = None) -> dbc.CardHeader:
    row = [html.Div(title, className="panel-title", style={"flex": "1"})]

    tooltip = None
    if info_id and info_text:
        icon = html.I("", className="fa-solid fa-circle-info panel-info-icon", id=info_id)
        row.append(icon)
        tooltip = dbc.Tooltip(
            info_text,
            target=info_id,
            placement="left",
            trigger="hover",
            autohide=True,
            delay={"show": 250, "hide": 0},
            style={"maxWidth": "420px"},
        )

    header_row = html.Div(row, className="panel-header-row")
    return dbc.CardHeader([header_row, tooltip] if tooltip else header_row, className="panel-header")


def kpi_textbox(component_id: str) -> html.Pre:
    """Unified text box style for KPI interpretations."""
    return html.Pre(
        id=component_id,
        className="meta-text",
        style={
            "whiteSpace": "pre-wrap",
            "margin": "0",
            "fontSize": "0.95rem",
            "lineHeight": "1.25rem",
        },
    )


# =============================================================================
# Helpers for sector -> load mapping
# =============================================================================
def build_sector_map(load_cols: List[str], summary_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    available_ids = {load_id(c) for c in load_cols}
    tmp = summary_df.copy()
    tmp = tmp[tmp["load_id"].isin(available_ids)].copy()

    sector_to_ids: Dict[str, List[str]] = (
        tmp.groupby("Sector_group")["load_id"].apply(lambda x: sorted(set(x.astype(str)))).to_dict()
    )

    loadcols_set = set(load_cols)
    sector_to_loadcols: Dict[str, List[str]] = {}
    for sector, ids in sector_to_ids.items():
        cols = [f"load_{i}" for i in ids]
        cols = [c for c in cols if c in loadcols_set]
        if cols:
            sector_to_loadcols[sector] = cols

    sectors = sorted(sector_to_loadcols.keys())
    return sectors, sector_to_loadcols


def make_empty_message_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=CFG.fig_height,
        margin=dict(l=60, r=20, t=60, b=55),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(color=TEXT)),
        showlegend=False,
    )
    return add_frame(fig)


# =============================================================================
# App factory
# =============================================================================
def make_app(df: pd.DataFrame, load_cols: List[str], summary_df: pd.DataFrame) -> Dash:
    external_stylesheets = [
        CFG.theme,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
    ]
    app = Dash(__name__, title=CFG.title, external_stylesheets=external_stylesheets)

    sectors, sector_to_loadcols = build_sector_map(load_cols, summary_df)
    if not sectors:
        raise RuntimeError(
            "Keine Sector_group-Zuordnung gefunden, die zu den vorhandenen load_* Spalten passt. "
            "Prüfe summary.csv (load_id) und Zeitreihen-CSV (load_* Spalten)."
        )

    default_sector = sectors[0]
    default_load = sector_to_loadcols[default_sector][0]

    ipa_logo = html.Img(src="/assets/path14.svg", className="navbar-logo", alt="Fraunhofer IPA")

    refresh_badge = dbc.Badge(
        [html.I(className="fa-solid fa-rotate-right me-2"), "Aktualisieren"],
        id="btn-refresh",
        color="secondary",
        pill=True,
        className="ms-2 badge-refresh",
    )

    selection_header = dbc.CardHeader(
        html.Div(
            className="panel-header-row",
            children=[
                html.Div("Auswahlmenü", className="panel-title", style={"flex": "1"}),
                dbc.Button(
                    html.I(id="icon-collapse-selection", className="fa-solid fa-chevron-up"),
                    id="btn-collapse-selection",
                    color="link",
                    className="p-0",
                    n_clicks=0,
                ),
            ],
        ),
        className="panel-header",
    )

    app.layout = html.Div(
        className="app-root",
        children=[
            dcc.Store(id="refresh-store"),
            dcc.Store(id="store-sector-map", data=sector_to_loadcols),

            dbc.Navbar(
                dbc.Container(
                    fluid=True,
                    className="navbar-inner",
                    children=[
                        html.Div(className="navbar-left", children=[ipa_logo, html.Div(CFG.title, className="navbar-title")]),
                        html.Div(
                            className="navbar-badges",
                            children=[
                                dbc.Badge("Fraunhofer IPA | 80603", color="secondary", className="ms-2"),
                                refresh_badge,
                            ],
                        ),
                    ],
                ),
                color="dark",
                dark=True,
                className="app-navbar shadow-sm",
            ),

            dbc.Container(
                fluid=True,
                className="page-container",
                children=[
                    # --- Auswahl (einklappbar) ---
                    dbc.Row(
                        className="mb-3",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        selection_header,
                                        dbc.Collapse(
                                            id="collapse-selection",
                                            is_open=True,
                                            children=dbc.CardBody(
                                                dbc.Row(
                                                    className="g-3",
                                                    children=[
                                                        dbc.Col(
                                                            md=4,
                                                            children=[
                                                                dbc.Label("Branche", className="form-label-dark"),
                                                                dcc.Dropdown(
                                                                    id="dd-sector",
                                                                    className="dash-dropdown",
                                                                    options=[{"label": s, "value": s} for s in sectors],
                                                                    value=default_sector,
                                                                    clearable=False,
                                                                    searchable=True,
                                                                ),
                                                            ],
                                                        ),
                                                        dbc.Col(
                                                            md=4,
                                                            children=[
                                                                dbc.Label("Zeitreihe", className="form-label-dark"),
                                                                dcc.Dropdown(
                                                                    id="dd-load",
                                                                    className="dash-dropdown",
                                                                    options=[{"label": label_for_load(c), "value": c} for c in sector_to_loadcols[default_sector]],
                                                                    value=default_load,
                                                                    clearable=False,
                                                                    searchable=True,
                                                                ),
                                                            ],
                                                        ),
                                                        dbc.Col(
                                                            md=4,
                                                            children=[
                                                                dbc.Label("Info", className="form-label-dark"),
                                                                html.Div(id="meta", className="meta-text"),
                                                            ],
                                                        ),
                                                    ],
                                                )
                                            ),
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),

                    # --- Plot 1: Time Series ---
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel("Visualisierung", info_id="info-ts", info_text="Zeitreihe der ausgewählten Last."),
                                        dbc.CardBody(dcc.Graph(id="g-ts", config=graph_config(), className="dash-graph")),
                                    ],
                                ),
                            )
                        ],
                    ),

                    # --- KPI Row 1: Self Similarity + Min/Max/Median ---
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                lg=6,
                                md=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm h-100",
                                    children=[
                                        panel(
                                            "Selbstähnlichkeit Zeitreihe",
                                            info_id="info-ss",
                                            info_text="Vergleich Target-Load gegen Referenzen (Lag 1d / 1w / Best-of).",
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="g-self-sim",
                                                    config=graph_config(),
                                                    className="dash-graph",
                                                    style={"height": "520px"},
                                                ),
                                                html.Hr(className="my-2"),
                                                kpi_textbox("txt-self-sim"),
                                            ]
                                        ),
                                    ],
                                ),
                            ),
                            dbc.Col(
                                lg=6,
                                md=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm h-100",
                                    children=[
                                        panel(
                                            "Einordnung der Last",
                                            info_id="info-mm",
                                            info_text="Min/Max-Band und Median über Referenzen (sortiert nach Median) mit hervorgehobener Target-Last.",
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="g-minmax",
                                                    config=graph_config(),
                                                    className="dash-graph",
                                                    style={"height": "520px"},
                                                ),
                                                html.Hr(className="my-2"),
                                                kpi_textbox("txt-minmax"),
                                            ]
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),

                    # --- KPI Row 2: Weather Dependence (full width) ---
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel(
                                            "Wetterabhängigkeit",
                                            info_id="info-wd",
                                            info_text="Spearman-Korrelation zwischen Last und wetterbasiertem linearem Modell (über saisonale Testmonate).",
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="g-weather",
                                                    config=graph_config(),
                                                    className="dash-graph",
                                                    style={"height": "520px"},
                                                ),
                                                html.Hr(className="my-2"),
                                                kpi_textbox("txt-weather"),
                                            ]
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    # --- Clientside reload callback (Badge click) ---
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks && n_clicks > 0) {
                window.location.reload();
            }
            return n_clicks;
        }
        """,
        Output("refresh-store", "data"),
        Input("btn-refresh", "n_clicks"),
        prevent_initial_call=True,
    )

    # --- Collapse Toggle für Auswahlmenü ---
    @app.callback(
        Output("collapse-selection", "is_open"),
        Output("icon-collapse-selection", "className"),
        Input("btn-collapse-selection", "n_clicks"),
        State("collapse-selection", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_selection_collapse(n_clicks: int, is_open: bool):
        new_open = not is_open
        icon = "fa-solid fa-chevron-up" if new_open else "fa-solid fa-chevron-down"
        return new_open, icon

    # --- Callback 1: Load-Dropdown nach Branche filtern ---
    @app.callback(
        Output("dd-load", "options"),
        Output("dd-load", "value"),
        Input("dd-sector", "value"),
        State("store-sector-map", "data"),
        prevent_initial_call=False,
    )
    def update_load_dropdown(selected_sector: str, sector_map: dict):
        if not selected_sector or not sector_map:
            return no_update, no_update

        cols = sector_map.get(selected_sector, [])
        options = [{"label": label_for_load(c), "value": c} for c in cols]
        new_value = cols[0] if cols else None
        return options, new_value

    # --- Callback 2: Plots aktualisieren (abhängig von Branche + Load) ---
    @app.callback(
        Output("g-ts", "figure"),
        Output("meta", "children"),
        Output("g-self-sim", "figure"),
        Output("txt-self-sim", "children"),
        Output("g-minmax", "figure"),
        Output("txt-minmax", "children"),
        Output("g-weather", "figure"),
        Output("txt-weather", "children"),
        Input("dd-sector", "value"),
        Input("dd-load", "value"),
        State("store-sector-map", "data"),
        prevent_initial_call=False,
    )
    def update_dashboard(selected_sector: str, load_col: str, sector_map: dict):
        if not load_col:
            empty = make_empty_message_figure("Keine Last ausgewählt.")
            return no_update, "Keine Last ausgewählt.", empty, "", empty, "", empty, ""

        load_col = str(load_col)

        if not selected_sector or not sector_map or selected_sector not in sector_map:
            empty = make_empty_message_figure("Keine gültige Branche ausgewählt.")
            return no_update, "Keine gültige Branche ausgewählt.", empty, "", empty, "", empty, ""

        sector_load_cols: List[str] = list(sector_map[selected_sector])
        if not sector_load_cols:
            empty = make_empty_message_figure(f"Branche '{selected_sector}' hat keine zugeordneten Loads.")
            return no_update, f"Branche '{selected_sector}' hat keine zugeordneten Loads.", empty, "", empty, "", empty, ""

        if load_col not in sector_load_cols:
            load_col = sector_load_cols[0]

        # ---- Plot 1: Zeitreihe ----
        fig_ts = fig_load_timeseries(df, load_col)

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        n = len(df)

        meta = (
            f"Branche: {selected_sector} • {label_for_load(load_col)} • Punkte: {n:,} • "
            f"Zeitraum: {ts_min:%Y-%m-%d %H:%M} → {ts_max:%Y-%m-%d %H:%M}"
        )

        # ---- KPIs nur innerhalb der Branche ----
        target_column = load_col
        reference_columns = [c for c in sector_load_cols if c != target_column]

        if not reference_columns:
            empty_fig = make_empty_message_figure("Nicht genug Referenzen in dieser Branche")
            msg = "Nicht genug Referenzen in dieser Branche."
            return fig_ts, meta, empty_fig, msg, empty_fig, msg, empty_fig, msg

        # IDs für ausgelagerte Module
        target_id = load_id(target_column)
        benchmark_ids = [load_id(c) for c in reference_columns]

        # ---- Plot 2: Self Similarity (ausgelagert) -> (fig, interpretation) ----
        sim_df = df[reference_columns + [target_column]].copy()
        for c in sim_df.columns:
            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce")
        sim_df = sim_df.dropna(axis=0, how="any")

        try:
            fig_selfsim, interpretation_selfsim = generate_self_similarity_plot(
                data_df=sim_df,
                reference_columns=reference_columns,
                target_column=target_column,
            )
        except Exception as e:
            LOGGER.exception("Self-similarity plot failed: %s", e)
            fig_selfsim = make_empty_message_figure("Selbstähnlichkeit: Fehler in Berechnung")
            interpretation_selfsim = "Selbstähnlichkeit konnte nicht berechnet werden (Fehler / Daten fehlen)."

        # ---- Plot 3: Min/Max/Median (ausgelagert) -> (fig, capture) ----
        try:
            fig_minmax, capture_minmax = generate_distribution_comparison(
                data_df=df,
                benchmark_columns=benchmark_ids,
                target_column=target_id,
            )
        except Exception as e:
            LOGGER.exception("Min/Max/Median plot failed: %s", e)
            fig_minmax = make_empty_message_figure("Einordnung der Last: Fehler in Berechnung")
            capture_minmax = "Einordnung konnte nicht berechnet werden (Fehler / Daten fehlen)."

        # ---- Plot 4: Weather Dependence (ausgelagert) -> (fig, capture) ----
        try:
            weather_out = generate_weather_dependence_plot(
                data_df=df,
                benchmark_columns=benchmark_ids,
                target_column=target_id,
            )
            if isinstance(weather_out, tuple):
                fig_weather, capture_weather = weather_out
            else:
                fig_weather, capture_weather = weather_out, ""
        except Exception as e:
            LOGGER.exception("Weather dependence plot failed: %s", e)
            fig_weather = make_empty_message_figure("Wetterabhängigkeit: Fehler in Berechnung / Features fehlen")
            capture_weather = "Wetterabhängigkeit konnte nicht berechnet werden (Features fehlen oder Fehler)."

        # Einheitliches Theme + Rahmen
        fig_ts = apply_dashboard_theme(fig_ts, showlegend=False)
        fig_selfsim = apply_dashboard_theme(fig_selfsim, showlegend=False)  # selfsim regelt showlegend intern
        fig_minmax = apply_dashboard_theme(fig_minmax)
        fig_weather = apply_dashboard_theme(fig_weather)

        fig_ts = add_frame(fig_ts)
        fig_selfsim = add_frame(fig_selfsim)
        fig_minmax = add_frame(fig_minmax)
        fig_weather = add_frame(fig_weather)

        return (
            fig_ts,
            meta,
            fig_selfsim,
            interpretation_selfsim,
            fig_minmax,
            capture_minmax,
            fig_weather,
            capture_weather,
        )

    return app


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    data_file = resolve_data_file()
    summary_file = resolve_summary_file()

    df = load_timeseries_csv(data_file)
    load_cols = find_load_columns(df)
    summary_df = load_summary_csv(summary_file)

    LOGGER.info("Gefundene Loads: %s", load_cols)
    LOGGER.info(
        "Rows=%s | Cols=%s | Range: %s .. %s",
        len(df),
        df.shape[1],
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    LOGGER.info("Summary rows: %s | unique sectors: %s", len(summary_df), summary_df["Sector_group"].nunique())

    app = make_app(df, load_cols, summary_df)
    app.run_server(host=CFG.host, port=CFG.port, debug=CFG.debug, use_reloader=False)


if __name__ == "__main__":
    main()
