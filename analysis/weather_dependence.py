# analysis/weather_dependence.py
# -----------------------------------------------------------------------------
# Weather dependence analysis (Plotly version)
# - Keeps existing computation logic (corr, filtering, feature selection)
# - Replaces Matplotlib plotting with Plotly for seamless Dash integration
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from helpers.weather_mapping import WEATHER_LABELS

# Colors (kept from original)
c_min = "#4B80B9"     # blue (reference scatter)
c_max = "#54A24B"     # not used here
c_med = "#F58518"     # not used here (we use violin median)
c_target = "#E45756"  # red (target line/markers)
c_band = "lightgrey"  # violin fill



def _pct_leq(values: pd.Series, x: float) -> float:
    """Perzentil-artig: Anteil der Referenzen <= x in %."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if v.size == 0 or not np.isfinite(x):
        return np.nan
    return 100.0 * (v <= float(x)).mean()

def _klasse_typisch(p: float, typisch_band=(40, 60), eher_band=(25, 75)) -> str:
    """Sehr einfache Einordnung über Perzentile."""
    if np.isnan(p):
        return "nicht berechenbar"
    lo_typ, hi_typ = typisch_band
    lo_eher, hi_eher = eher_band
    if lo_typ <= p <= hi_typ:
        return "typisch"
    if p < lo_eher:
        return "deutlich niedriger"
    if p < lo_typ:
        return "eher niedriger"
    if p > hi_eher:
        return "deutlich höher"
    if p > hi_typ:
        return "eher höher"
    return "typisch"

def _richtung(p: float) -> str:
    """Ob target eher 'höher' oder 'niedriger' als Referenzen liegt."""
    if np.isnan(p):
        return "unbekannt"
    return "höher" if p > 50 else "niedriger" if p < 50 else "ähnlich"

def describe_weather_dependency_de(
    all_corr: pd.DataFrame,
    target_col: str,
    weather_vars: list[str],
    overall_name: str = "Gesamtwetterabhängigkeit",
    typisch_band=(40, 60),
    eher_band=(25, 75),
    trend_threshold: float = 70.0,  # z.B. 70% der Variablen zeigen gleiche Richtung
) -> str:
    """
    Erzeugt eine kurze Beschreibung:
    1) Trend über Wettervariablen (ohne Overall)
    2) 2 stärkste Abweichungen (gegenüber Referenzen)
    3) 1 Satz zum Overall-Maß
    """

    labels = WEATHER_LABELS
    df = all_corr.copy()

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' nicht in all_corr.columns")

    # 1) Wettervariablen ohne Overall auswählen
    vars_main = [v for v in weather_vars if v in df.index]
    if overall_name in vars_main:
        vars_main.remove(overall_name)

    # Referenzen = alle Spalten außer target
    ref_cols = [c for c in df.columns if c != target_col]
    if not ref_cols:
        raise ValueError("Es werden Referenzprofile benötigt (mind. eine Spalte neben target).")

    # Perzentile je Variable berechnen
    rows = []
    for v in vars_main:
        ref_vals = df.loc[v, ref_cols]
        t = float(df.loc[v, target_col])
        p = _pct_leq(ref_vals, t)  # Anteil refs <= target
        rows.append((v, t, p))

    dep = pd.DataFrame(rows, columns=["var", "target", "pct"]).set_index("var")

    # Trendanalyse: wie viele liegen >50 bzw. <50 bzw. typisch
    n = len(dep)
    if n == 0:
        trend_text = "Es liegen keine Wettervariablen zur Auswertung vor."
        top_text = ""
    else:
        n_higher = int((dep["pct"] > 50).sum())
        n_lower  = int((dep["pct"] < 50).sum())
        n_equal  = n - n_higher - n_lower
        share_higher = 100 * n_higher / n
        share_lower  = 100 * n_lower / n

        # Dominante Richtung?
        if share_higher >= trend_threshold:
            trend_text = (f"Über {share_higher:.0f}% der Wettervariablen zeigen beim Zielprofil "
                          f"eine höhere Abhängigkeit als die Referenzlastprofile.")
        elif share_lower >= trend_threshold:
            trend_text = (f"Über {share_lower:.0f}% der Wettervariablen zeigen beim Zielprofil "
                          f"eine geringere Abhängigkeit als die Referenzlastprofile.")
        else:
            # Gemischt
            if n_equal > 0:
                trend_text = (f"Die Wetterabhängigkeit des Zielprofils ist variablenabhängig: "
                              f"{share_higher:.0f}% der Variablen zeigen eine höhere, "
                              f"{share_lower:.0f}% eine geringere und "
                              f"{100 * n_equal / n:.0f}% eine ähnlich Abhängigket "
                              f"im Vergleich zu den Referenzen.")
            else:
                trend_text = (f"Die Wetterabhängigkeit des Zielprofils ist variablenabhängig: "
                              f"{share_higher:.0f}% der Variablen zeigen eine höhere und "
                              f"{share_lower:.0f}% eine geringere Abhängigket "
                              f"im Vergleich zu den Referenzen.")

        # 2) Zwei stärkste Abweichungen auswählen
        # Stärke = Abstand vom 50%-Perzentil (wie untypisch ist die Position)
        dep["dist50"] = (dep["pct"] - 50).abs()
        top2 = dep.sort_values("dist50", ascending=False).head(2)

        def fmt_line(v, t, p):
            name = labels.get(v, v)
            k = _klasse_typisch(p, typisch_band=typisch_band, eher_band=eher_band)
            # formuliere Richtung + Perzentil
            return (f"- {name}: Perzentil {p:.1f} "
                    f"({k} gegenüber Referenzen)")

        top_lines = [fmt_line(v, row["target"], row["pct"]) for v, row in top2.iterrows()]
        top_text = "Die stärksten Abweichungen zeigen:\n" + "\n".join(top_lines)

    # 3) Overall weather dependence Satz (optional)
    overall_text = ""
    if overall_name is not None and overall_name in df.index:
        # gleicher Perzentil-Vergleich, aber nur für overall
        ref_vals = df.loc[overall_name, ref_cols]
        t = float(df.loc[overall_name, target_col])
        p = _pct_leq(ref_vals, t)
        k = _klasse_typisch(p, typisch_band=typisch_band, eher_band=eher_band)

        overall_label = labels.get(overall_name, "Overall weather dependence")
        overall_text = (f"{overall_label}: Perzentil {p:.1f} "
                        f"({k} im Vergleich zu den Referenzen).")

    # Zusammenbauen
    parts = [trend_text]
    if top_text:
        parts.append(top_text)
    if overall_text:
        parts.append(overall_text)

    return "\n\n".join(parts)


##################################################################################


def weather_corr(df: pd.DataFrame, load_id: str, weather_vars: List[str], test_months: List[int] = [1, 4, 7, 11]) -> float:
    """
    Overall weather dependence measured as Spearman correlation between
    observed load and predictions of a weather-only linear model, evaluated
    on representative periods from multiple seasons to avoid seasonal bias.
    """
    y = df[f"load_{load_id}"].to_numpy()
    X = df[[f"{v}_{load_id}" for v in weather_vars]].to_numpy()

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]

    test_idx = df[mask][f"month_{load_id}"].isin(test_months)

    X_train, X_test = X[~test_idx], X[test_idx]
    y_train, y_test = y[~test_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    return pd.Series(y).corr(pd.Series(y_hat), method="spearman")


def plot_dependency_violinplot(
    all_corr: pd.DataFrame,
    *,
    target_col: str,
    target_column: str,
    title: str = "Wetterabhängigkeit der Last",
) -> go.Figure:
    """
    Plotly violinplot of dependency values per variable across reference load profiles,
    with target profile highlighted.

    Parameters
    ----------
    all_corr : DataFrame
        Rows = variables, columns = load profiles.
    target_col : str
        Column name of target load profile (e.g. "load_640").
    target_column : str
        Target ID string (e.g. "640") – used for label.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    df = all_corr.copy()

    # Reference columns = all except target
    ref_cols = [c for c in df.columns if c != target_col]

    # Stable order for x categories
    vars_order = list(df.index)

    # Violin data (one trace per variable)
    fig = go.Figure()

    # Baseline y=0
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(vars_order) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="#9CA3AF", width=2, dash="dot"),
        layer="below",
    )

    rng = np.random.default_rng(42)

    # Add violins + reference jitter points
    for i, var in enumerate(vars_order):
        vals = df.loc[var, ref_cols].dropna().values.astype(float)

        # Violin (distribution)
        fig.add_trace(
            go.Violin(
                x=np.full(len(vals), i),
                y=vals,
                name=str(var),
                showlegend=False,
                points=False,
                box_visible=False,
                meanline_visible=False,
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(220,220,220,0.20)",  # lightgrey with alpha
                opacity=1.0,
                spanmode="hard",
                width=0.8,
            )
        )

        # Reference scatter inside the violin with jitter on x
        if len(vals) > 0:
            jitter = rng.normal(loc=0.0, scale=0.08, size=len(vals))
            fig.add_trace(
                go.Scatter(
                    x=np.full(len(vals), i) + jitter,
                    y=vals,
                    mode="markers",
                    marker=dict(size=6, color=c_min, opacity=0.6),
                    showlegend=False,
                    hovertemplate="Ref: %{y:.3f}<extra></extra>",
                )
            )

    # Target series across variables
    target_vals = df[target_col].values.astype(float)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(vars_order)),
            y=target_vals,
            mode="markers",
            name=f"Target {target_column}",
            line=dict(color=c_target, width=2),
            marker=dict(symbol="diamond", size=10, color=c_target),
            hovertemplate="Target: %{y:.3f}<extra></extra>",
            showlegend=True,
        )
    )

    # X labels (pretty names)
    x_labels = [WEATHER_LABELS.get(v, v) for v in vars_order]

        # --- dynamic y range (robust) ---
    vals_ref = df[ref_cols].to_numpy().astype(float).ravel()
    vals_tgt = df[target_col].to_numpy().astype(float).ravel()
    vals = np.concatenate([vals_ref, vals_tgt])
    vals = vals[np.isfinite(vals)]

    if vals.size:
        q_low, q_high = np.quantile(vals, [0.05, 0.95])  # robust against outliers
        span = max(q_high - q_low, 1e-6)
        pad = 0.15 * span

        lo = q_low - pad
        hi = q_high + pad

        # Optional: symmetrisch um 0 (macht Korrelationen leichter vergleichbar)
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m

        # Hard clamp (Korrelation liegt in [-1,1])
        lo = max(lo, -1.0)
        hi = min(hi, 1.0)
    else:
        lo, hi = -1.0, 1.0


    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(
            title="Wettervariable",
            tickmode="array",
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            tickangle=0,
        ),
        yaxis=dict(
            title="Spearman-Korrelation (ρ)",
            range=[lo, hi],
            zeroline=False,
        ),
        margin=dict(l=60, r=30, t=60, b=90),
        height=520,
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.08,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


def generate_distribution_comparison(data_df: pd.DataFrame, benchmark_columns: List[str], target_column: str) -> go.Figure:
    """
    Entry point used by the dashboard.

    Parameters
    ----------
    data_df : DataFrame
        Must contain:
          - load_{id} for each id in benchmark_columns + target_column
          - weather vars per id: {var}_{id} for selected vars
          - month_{id} for overall dependence calculation
    benchmark_columns : List[str]
        Reference IDs (without "load_").
    target_column : str
        Target ID (without "load_").

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    ref_cols = [f"load_{x}" for x in benchmark_columns]
    target_col = f"load_{target_column}"

    # Base set from original
    weather_vars = ["temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]

    # Drop weather vars with high proportion of missing values (> 30%)
    miss_prop = data_df.isna().sum() / max(len(data_df), 1)
    high_missing = miss_prop[miss_prop > 0.3].index

    # Convert "temp_640" -> "temp"
    base_vars_high_missing = high_missing.to_series().astype(str).str.rsplit("_", n=1).str[0]
    weather_vars_to_drop = sorted(set(base_vars_high_missing).intersection(weather_vars))
    weather_vars = [v for v in weather_vars if v not in weather_vars_to_drop]

    all_cols = ref_cols + [target_col]
    all_col_names = benchmark_columns + [target_column]

    # Spearman correlation per load, per weather var
    all_corr_series = []
    for i in range(len(all_cols)):
        load_name = all_cols[i]          # e.g. "load_640"
        load_id = all_col_names[i]       # e.g. "640"
        weather_feature_cols = [f"{v}_{load_id}" for v in weather_vars]

        # Correlate load with its weather features
        corr = data_df[[load_name] + weather_feature_cols].corr(method="spearman")
        corr = corr[load_name].drop(load_name)  # series indexed by weather_feature_cols
        corr.index = weather_vars               # normalize index to base var names
        all_corr_series.append(corr.rename(load_name))

    all_corr = pd.concat(all_corr_series, axis=1)  # rows=vars, cols=load_*
    all_corr = all_corr.dropna(how="all")

    # Overall dependence: spearman(y, y_hat(weather-only))
    all_r2 = []
    for col_id in all_col_names:
        all_r2.append(weather_corr(data_df, col_id, weather_vars=list(all_corr.index)))

    all_corr.loc["Overall Weather Dependence", :] = all_r2

    fig = plot_dependency_violinplot(
        all_corr,
        target_col=target_col,
        target_column=target_column,
        title="Wetterabhängigkeit der Last",
    )

    # Create figure caption
    capture = describe_weather_dependency_de(all_corr, target_col, weather_vars)

    return fig, capture
