# app.py
# -----------------------------------------------------------------------------
# Plotly Dash Grundgerüst – Styling via assets/styles.css
# - CSV: 4_loads.csv
# - Dropdown zur Auswahl der Last (load_*)
# - Zeitreihen-Plot unter dem Dropdown
# - Refresh Badge rechts in der Navbar (Client-side Reload)
# - IPA-Logo (assets/ipa.svg) links in der Title Bar, korrekt skaliert via CSS
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
def project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


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

    # Robust: erste Spalte als Timestamp (typischerweise "Unnamed: 0")
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


def find_load_columns(df: pd.DataFrame) -> List[str]:
    load_cols = [c for c in df.columns if c.startswith("load_")]
    if not load_cols:
        raise KeyError("Keine Spalten gefunden, die mit 'load_' beginnen.")
    return sorted(load_cols)


def label_for_load(col: str) -> str:
    return col.replace("load_", "Load ")


# =============================================================================
# Figure styling (Layout über CSS, Plot über Plotly)
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

    # --- Logo links (assets/path14.svg wird automatisch unter /assets/ ausgeliefert) ---
    ipa_logo = html.Img(
        src="/assets/path14.svg",
        className="navbar-logo",
        alt="Fraunhofer IPA",
    )

    # --- Refresh Badge (rechts in der Navbar) ---
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
            # Dummy store for clientside page reload
            dcc.Store(id="refresh-store"),

            dbc.Navbar(
                dbc.Container(
                    fluid=True,
                    className="navbar-inner",
                    children=[
                        # LEFT: Logo + Title
                        html.Div(
                            className="navbar-left",
                            children=[
                                ipa_logo,
                                html.Div(CFG.title, className="navbar-title"),
                            ],
                        ),
                        # RIGHT: Badges + Refresh
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
                                                        children=[
                                                            dbc.Label("Last", className="form-label-dark"),
                                                            dcc.Dropdown(
                                                                id="dd-load",
                                                                className="dash-dropdown",
                                                                options=[{"label": label_for_load(c), "value": c} for c in load_cols],
                                                                value=default_load,
                                                                clearable=False,
                                                                searchable=False,
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

                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel("Visualisierung", info_id="info-ts", info_text="TO BE DONE"),
                                        dbc.CardBody(
                                            dcc.Graph(id="g-ts", config=graph_config(), className="dash-graph"),
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

    @app.callback(
        Output("g-ts", "figure"),
        Output("meta", "children"),
        Input("dd-load", "value"),
        prevent_initial_call=False,
    )
    def update_timeseries(load_col: str):
        load_col = str(load_col)
        fig = fig_load_timeseries(df, load_col)

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        n = len(df)

        meta = f"{label_for_load(load_col)} • Punkte: {n:,} • Zeitraum: {ts_min:%Y-%m-%d %H:%M} → {ts_max:%Y-%m-%d %H:%M}"
        return fig, meta

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
