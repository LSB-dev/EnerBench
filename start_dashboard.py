# app.py
# -----------------------------------------------------------------------------
# Plotly Dash Grundgerüst – Styling via assets/styles.css
# - CSV: 4_loads.csv
# - Dropdown zur Auswahl der Last (load_*)
# - Plot 1 (Plotly): Zeitreihe der gewählten Last
# - Plot 2 (Matplotlib -> PNG): Self-Similarity via analysis/self_similarity/run_self_similarity.py
# - Plot 3 (Plotly): Min/Max/Median Distribution Comparison via analysis/min_max_median.py
#       * Re-use der bestehenden Funktion generate_distribution_comparison(...)
#       * Rendering im Dashboard als Plotly (Konvertierung Matplotlib -> Plotly-Daten)
# - Refresh Badge rechts in der Navbar (Client-side Reload)
# - IPA-Logo (assets/path14.svg)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import logging
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

# Matplotlib nur für: Self-Similarity Render + Close() + Konvertierung min_max_median -> plotly
from matplotlib import pyplot as plt


# =============================================================================
# Ensure project root in sys.path (for imports like analysis.*)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Self-similarity (existing logic)
from analysis.self_similarity.run_self_similarity import generate_self_similarity_plot  # noqa: E402

# Distribution comparison (existing logic, matplotlib)
from analysis.min_max_median import generate_distribution_comparison  # noqa: E402


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
    fig_height_small: int = 420


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

    # Robust: erste Spalte als Timestamp
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


def id_for_load(col: str) -> str:
    # "load_640" -> "640"
    return col.replace("load_", "")


# =============================================================================
# Plotly styling
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


def style_figure(fig: go.Figure, title: str, x_label: str, y_label: str, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, Segoe UI, Arial", size=12, color=TEXT),
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", font=dict(color=TEXT)),
        autosize=True,
        height=height,
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
    fig = style_figure(fig, "Zeitreihe", "Zeit", "Leistung (kW)", height=CFG.fig_height)
    fig.update_layout(showlegend=False)
    return add_frame(fig)


# =============================================================================
# Matplotlib Figure -> Base64 PNG (für Dash html.Img)
# =============================================================================
def mpl_fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# =============================================================================
# Matplotlib -> Plotly conversion (for min_max_median plot)
# We intentionally re-use the existing min_max_median.generate_distribution_comparison(...)
# and only convert its rendered data into a Plotly figure for dashboard visualization.
# =============================================================================
def plotly_from_distribution_mpl(fig_mpl) -> go.Figure:
    """
    Convert the matplotlib figure created by generate_distribution_comparison(...)
    into a Plotly figure (lines + band + highlighted target).
    """
    ax = fig_mpl.axes[0]

    # Extract line data by label ("min", "max", "median")
    lines_by_label: Dict[str, Any] = {}
    for ln in ax.lines:
        label = (ln.get_label() or "").strip()
        if label in {"min", "max", "median"}:
            lines_by_label[label] = ln

    if not {"min", "max", "median"}.issubset(lines_by_label.keys()):
        raise RuntimeError("Konnte die erwarteten Linien ('min','max','median') aus dem Matplotlib-Plot nicht extrahieren.")

    x = list(lines_by_label["min"].get_xdata())
    y_min = list(lines_by_label["min"].get_ydata())
    y_max = list(lines_by_label["max"].get_ydata())
    y_med = list(lines_by_label["median"].get_ydata())

    # Target x-position from xticks (the function sets exactly one tick at target)
    xticks = ax.get_xticks()
    x_target = float(xticks[0]) if len(xticks) else None

    # Target points from PathCollection offsets (scatter creates a PathCollection)
    # We'll pick the collection with 3 points (min/median/max).
    target_points = None
    for coll in getattr(ax, "collections", []):
        if hasattr(coll, "get_offsets"):
            offsets = coll.get_offsets()
            # offsets can be a numpy array; length check is robust
            try:
                if len(offsets) == 3:
                    target_points = offsets
                    break
            except Exception:
                continue

    # Build Plotly figure
    fig = go.Figure()

    # Band (min..max)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_min,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="band_min",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_max,
            mode="lines",
            fill="tonexty",
            opacity=0.25,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="band_max",
        )
    )

    # Lines
    fig.add_trace(go.Scatter(x=x, y=y_min, mode="lines", name="min", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_max, mode="lines", name="max", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_med, mode="lines", name="median", line=dict(width=2)))

    # Highlight target: dashed vertical + points
    shapes = []
    if x_target is not None and target_points is not None:
        # offsets: [[x, y_min],[x, y_med],[x, y_max]] (order as plotted)
        ys = sorted([float(p[1]) for p in target_points])
        y_tmin, y_tmed, y_tmax = ys[0], ys[1], ys[2]

        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=x_target,
                x1=x_target,
                y0=y_tmin,
                y1=y_tmax,
                line=dict(width=2, dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_target, x_target, x_target],
                y=[y_tmin, y_tmed, y_tmax],
                mode="markers",
                name="target",
                marker=dict(size=9),
            )
        )

        # Show only target tick label (as original mpl)
        ticktext = [str(ax.get_xticklabels()[0].get_text())] if ax.get_xticklabels() else ["target"]
        fig.update_xaxes(tickmode="array", tickvals=[x_target], ticktext=ticktext)

    # Titles/labels from mpl
    title = ax.get_title() or "Distribution Comparison"
    ylab = ax.get_ylabel() or "Load / kW"

    fig = style_figure(fig, title, "Profil-Index (sortiert nach Median)", ylab, height=CFG.fig_height_small)
    fig = add_frame(fig)
    return fig


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
                            children=[
                                ipa_logo,
                                html.Div(CFG.title, className="navbar-title"),
                            ],
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
                    # --- Auswahl ---
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

                    # --- Plot 1: Zeitreihe ---
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel("Zeitreihe", info_id="info-ts", info_text="Zeitreihe der ausgewählten Last."),
                                        dbc.CardBody(
                                            dcc.Graph(id="g-ts", config=graph_config(), className="dash-graph"),
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),

                    # --- Plot 2: Self Similarity (Matplotlib -> PNG) ---
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel(
                                            "Self-Similarity",
                                            info_id="info-ss",
                                            info_text="Vergleich Target-Load gegen alle anderen Loads (Lag 1d / 1w / Best-of).",
                                        ),
                                        dbc.CardBody(
                                            dcc.Loading(
                                                type="default",
                                                children=html.Img(
                                                    id="img-self-sim",
                                                    style={"width": "100%", "height": "auto"},
                                                    alt="Self Similarity Plot",
                                                ),
                                            )
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),

                    # --- Plot 3: Min/Max/Median Distribution Comparison (Plotly) ---
                    dbc.Row(
                        className="mb-4",
                        children=[
                            dbc.Col(
                                width=12,
                                children=dbc.Card(
                                    className="card-surface shadow-sm",
                                    children=[
                                        panel(
                                            "Min/Max/Median Vergleich",
                                            info_id="info-dist",
                                            info_text="Min/Max-Band und Median über alle Lastprofile (sortiert nach Median) mit hervorgehobenem Target.",
                                        ),
                                        dbc.CardBody(
                                            dcc.Loading(
                                                type="default",
                                                children=dcc.Graph(
                                                    id="g-dist",
                                                    config=graph_config(),
                                                    className="dash-graph",
                                                ),
                                            )
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
        Output("img-self-sim", "src"),
        Output("g-dist", "figure"),
        Input("dd-load", "value"),
        prevent_initial_call=False,
    )
    def update_all(load_col: str):
        load_col = str(load_col)

        # ---- Plot 1: Time Series (Plotly) ----
        fig_ts = fig_load_timeseries(df, load_col)

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        n = len(df)
        meta = f"{label_for_load(load_col)} • Punkte: {n:,} • Zeitraum: {ts_min:%Y-%m-%d %H:%M} → {ts_max:%Y-%m-%d %H:%M}"

        # Prepare numeric-only DF for analysis functions (avoid strings)
        # NOTE: keep only load columns needed; timestamp not required
        target_column = load_col
        reference_columns = [c for c in load_cols if c != target_column]

        numeric_df = df[reference_columns + [target_column]].copy()
        for c in numeric_df.columns:
            numeric_df[c] = pd.to_numeric(numeric_df[c], errors="coerce")

        # ---- Plot 2: Self Similarity (existing logic, strict no-NaN requirement) ----
        fig_mpl_ss = None
        try:
            ss_df = numeric_df.dropna(axis=0, how="any")
            fig_mpl_ss = generate_self_similarity_plot(
                data_df=ss_df,
                reference_columns=reference_columns,
                target_column=target_column,
            )
            img_self_sim = mpl_fig_to_base64_png(fig_mpl_ss)
        finally:
            if fig_mpl_ss is not None:
                try:
                    plt.close(fig_mpl_ss)
                except Exception:
                    pass

        # ---- Plot 3: Distribution Comparison (existing logic, then Plotly visualization) ----
        # min_max_median expects ids without "load_" prefix as strings, and internally builds load_*
        benchmark_ids = [id_for_load(c) for c in reference_columns]  # all other loads
        target_id = id_for_load(target_column)

        fig_mpl_dist = None
        try:
            fig_mpl_dist = generate_distribution_comparison(
                data_df=df,  # function works with NaNs (pandas min/median/max ignore NaNs); keep full df
                benchmark_columns=benchmark_ids,
                target_column=target_id,
            )
            fig_dist = plotly_from_distribution_mpl(fig_mpl_dist)
        finally:
            if fig_mpl_dist is not None:
                try:
                    plt.close(fig_mpl_dist)
                except Exception:
                    pass

        return fig_ts, meta, img_self_sim, fig_dist

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
