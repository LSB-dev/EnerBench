# app.py
# -----------------------------------------------------------------------------
# Plotly Dash Grundgerüst – Styling via assets/styles.css
# - CSV: 4_loads.csv
# - Dropdown zur Auswahl der Last (load_*)
# - Zeitreihen-Plot
# - Self-Similarity Plot (Plotly, basiert auf eurer Logik: sMAPE, lag 1d & 1w)
# - Refresh Badge rechts in der Navbar (Client-side Reload)
# - IPA-Logo (assets/path14.svg) links in der Title Bar
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc


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
DEFAULT_DATA_FILE = Path(r"D:\EnerBench\RawData\4_loads.csv")


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
        "Keine CSV gefunden.\n"
        "Lösung:\n"
        "  1) Lege 4_loads.csv neben app.py\n"
        "  ODER\n"
        "  2) setze ENV DATA_FILE auf den Pfad zur CSV\n"
        f"Default geprüft: {DEFAULT_DATA_FILE}"
    )


def load_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Lade CSV: %s", path)
    df = pd.read_csv(path, low_memory=False)

    # Erste Spalte als Timestamp (bei euch oft leer benannt / Unnamed: 0)
    time_col = df.columns[0]
    df = df.rename(columns={time_col: "timestamp"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    before = len(df)
    df = df[df["timestamp"].notna()].copy()
    removed = before - len(df)
    if removed:
        LOGGER.info("Rows ohne gültigen Timestamp entfernt: %s", removed)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def find_load_columns(df: pd.DataFrame) -> List[str]:
    load_cols = [c for c in df.columns if str(c).startswith("load_")]
    if not load_cols:
        raise KeyError("Keine Spalten gefunden, die mit 'load_' beginnen.")
    # numerisch sortieren: load_116 < load_665
    return sorted(load_cols, key=lambda c: int(str(c).split("_")[1]))


def label_for_load(col: str) -> str:
    return str(col).replace("load_", "Load ")


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


def style_figure(fig: go.Figure, title: str, x_label: str, y_label: str, *, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, Segoe UI, Arial", size=12, color=TEXT),
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", font=dict(color=TEXT)),
        autosize=True,
        height=height or CFG.fig_height,
        margin=dict(l=60, r=20, t=60, b=55),
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


# =============================================================================
# Self similarity (Plotly version of your logic)
# =============================================================================
def sMAPE(y: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = (np.abs(y) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    num = np.abs(y - y_pred)
    return float(100.0 * np.mean(num[mask] / denom[mask]))


def _lag_scores(df_num: pd.DataFrame, lags: int) -> Dict[str, float]:
    shifted = df_num.shift(lags)
    # align: ignore first `lags`
    original = df_num.iloc[lags:]
    shifted = shifted.iloc[lags:]

    scores: Dict[str, float] = {}
    for col in original.columns:
        y = original[col].to_numpy()
        y_pred = shifted[col].to_numpy()
        scores[col] = sMAPE(y, y_pred)
    return scores


def compute_self_similarity_best(df: pd.DataFrame, cols: List[str], *, lag_1d: int = 96, lag_1w: int = 96 * 7) -> Dict[str, float]:
    """
    Best-of (min) zwischen lag 1 day und lag 1 week (wie in eurem Code).
    Erwartet: cols sind numerisch konvertierbar, NaNs werden vorher entfernt.
    """
    df_num = df[cols].apply(pd.to_numeric, errors="coerce")

    # robust: rows mit NaN in irgendeiner betrachteten Spalte rauswerfen
    df_num = df_num.dropna(axis=0, how="any")
    if len(df_num) <= max(lag_1d, lag_1w) + 1:
        raise ValueError(f"Zu wenig Daten nach NaN-Filter: rows={len(df_num)} für lags={max(lag_1d, lag_1w)}")

    s1 = _lag_scores(df_num, lags=lag_1d)
    s7 = _lag_scores(df_num, lags=lag_1w)
    return {c: min(s1[c], s7[c]) for c in s1.keys()}


def fig_self_similarity(df: pd.DataFrame, target_col: str, load_cols: List[str]) -> go.Figure:
    cols = [c for c in load_cols]  # include target too (for its own score)
    scores = compute_self_similarity_best(df, cols, lag_1d=96, lag_1w=96 * 7)

    reference_cols = [c for c in load_cols if c != target_col]
    ref_scores = [scores[c] for c in reference_cols]
    target_score = scores[target_col]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=ref_scores,
            nbinsx=35,
            name="Referenzen",
            opacity=0.95,
        )
    )

    # vertikale Linie für Target
    fig.add_vline(
        x=target_score,
        line_width=3,
        line_dash="solid",
        line_color="red",
    )

    fig.add_annotation(
        x=target_score,
        y=1,
        xref="x",
        yref="paper",
        text=f"<b>{label_for_load(target_col)}</b><br>sMAPE={target_score:.2f}%",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color=TEXT),
    )

    fig = style_figure(fig, "Self Similarity (best of 1d / 1w)", "sMAPE [%] (kleiner = ähnlicher)", "Count", height=420)
    fig.update_layout(bargap=0.06, showlegend=False)
    return add_frame(fig)


# =============================================================================
# Time series plot
# =============================================================================
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


# =============================================================================
# App factory
# =============================================================================
def make_app(df: pd.DataFrame, load_cols: List[str]) -> Dash:
    external_stylesheets = [
        CFG.theme,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
    ]
    app = Dash(__name__, title=CFG.title, external_stylesheets=external_stylesheets)

    default_load = load_cols[0]

    ipa_logo = html.Img(
        src="/assets/path14.svg",
        className="navbar-logo",
        alt="Fraunhofer IPA",
    )

    refresh_badge = dbc.Badge(
        [html.I(className="fa-solid fa-rotate-right me-2"), "Aktualisieren"],
        id="btn-refresh",
        color="secondary",
        pill=True,
        className="ms-2 badge-refresh",
    )

    app.layout = html.Div(
        className="app-root",
        children=[
            dcc.Store(id="refresh-store"),

            dbc.Navbar(
                dbc.Container(
                    fluid=True,
                    className="navbar-inner",
                    children=[
                        html.Div(
                            className="navbar-left",
                            children=[ipa_logo, html.Div(CFG.title, className="navbar-title")],
                        ),
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
                    # ------------------------
                    # Auswahl
                    # ------------------------
                    dbc.Row(
                        className="mb-3",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel("Auswahlmenü"),
                                        dbc.CardBody(
                                            dbc.Row(
                                                className="g-3",
                                                children=[
                                                    dbc.Col(
                                                        md=6,
                                                        # z-index Fix, falls Dropdown überlagert wird
                                                        style={"position": "relative", "zIndex": 5000},
                                                        children=[
                                                            dbc.Label("Last", className="form-label-dark"),
                                                            html.Div(
                                                                className="dropdown-wrap",
                                                                style={"position": "relative", "zIndex": 5000},
                                                                children=[
                                                                    dcc.Dropdown(
                                                                        id="dd-load",
                                                                        className="dash-dropdown",
                                                                        options=[{"label": label_for_load(c), "value": c} for c in load_cols],
                                                                        value=default_load,
                                                                        clearable=False,
                                                                        searchable=True,
                                                                        maxHeight=360,
                                                                        optionHeight=36,
                                                                        style={"zIndex": 5000},
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    dbc.Col(
                                                        md=6,
                                                        children=[
                                                            dbc.Label("Info", className="form-label-dark"),
                                                            html.Div(id="meta", className="meta-text"),
                                                        ],
                                                    ),
                                                ],
                                            )
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),

                    # ------------------------
                    # Zeitreihe
                    # ------------------------
                    dbc.Row(
                        className="mb-3",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel("Visualisierung – Zeitreihe", info_id="info-ts", info_text="Zeitreihe der ausgewählten Last."),
                                        dbc.CardBody(dcc.Graph(id="g-ts", config=graph_config(), className="dash-graph")),
                                    ],
                                ),
                            )
                        ],
                    ),

                    # ------------------------
                    # Self Similarity
                    # ------------------------
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel(
                                            "Self Similarity",
                                            info_id="info-ss",
                                            info_text="Histogramm der Self-Similarity (best of 1d/1w) aller Referenz-Loads; rote Linie = ausgewählter Target-Load.",
                                        ),
                                        dbc.CardBody(dcc.Graph(id="g-ss", config=graph_config(), className="dash-graph")),
                                    ],
                                ),
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    # Refresh: clientside reload
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

    # Main callback: update both plots + meta
    @app.callback(
        Output("g-ts", "figure"),
        Output("g-ss", "figure"),
        Output("meta", "children"),
        Input("dd-load", "value"),
        prevent_initial_call=False,
    )
    def update_dashboard(load_col: str):
        load_col = str(load_col)

        fig_ts = fig_load_timeseries(df, load_col)

        # Self similarity can fail if too few rows after NaN dropping -> handle gracefully
        try:
            fig_ss = fig_self_similarity(df, load_col, load_cols)
        except Exception as e:
            LOGGER.warning("Self-similarity failed: %s", e)
            fig_ss = style_figure(go.Figure(), "Self Similarity", "", "", height=420)
            fig_ss.add_annotation(
                text=f"<b>Self Similarity nicht verfügbar</b><br>{str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=TEXT, size=14),
            )
            fig_ss = add_frame(fig_ss)

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        n = len(df)
        meta = f"{label_for_load(load_col)} • Punkte: {n:,} • Zeitraum: {ts_min:%Y-%m-%d %H:%M} → {ts_max:%Y-%m-%d %H:%M}"

        return fig_ts, fig_ss, meta

    return app


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    data_file = resolve_data_file()
    df = load_csv(data_file)
    load_cols = find_load_columns(df)

    LOGGER.info("Gefundene Loads: %s", load_cols)
    LOGGER.info(
        "Rows=%s | Cols=%s | Range: %s .. %s",
        len(df),
        df.shape[1],
        df["timestamp"].min(),
        df["timestamp"].max(),
    )

    app = make_app(df, load_cols)
    app.run_server(host=CFG.host, port=CFG.port, debug=CFG.debug, use_reloader=False)


if __name__ == "__main__":
    main()
