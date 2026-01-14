# analysis/weather_dependence.py
# -----------------------------------------------------------------------------
# Weather dependence analysis (Plotly version)
# - Keeps existing computation logic (corr, filtering, feature selection)
# - Replaces Matplotlib plotting with Plotly for seamless Dash integration
# - Text: avoids misleading "Alle Wettervariablen ..." when only 1 variable exists
#        and adds a bridging sentence if overall contradicts single-variable trend
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
    v = np.abs(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
    x = np.abs(x)
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
        return "deutlich schwächer"
    if p < lo_typ:
        return "eher schwächer"
    if p > hi_eher:
        return "deutlich stärker"
    if p > hi_typ:
        return "eher stärker"
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
    overall_name: str = "Gesamt-Wetterabhängigkeit",
    typisch_band=(40, 60),
    eher_band=(25, 75),
    trend_threshold: float = 70.0,  # z.B. 70% der Variablen zeigen gleiche Richtung
) -> str:
    """
    Erzeugt eine kurze Beschreibung:
    1) Trend über Wettervariablen (ohne Overall)
    2) 2 stärkste Abweichungen (gegenüber Referenzen)
    3) 1 Satz zum Overall-Maß + optional Brückensatz bei scheinbarem Widerspruch
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

    # Perzentile je Variable berechnen (nur gültige Werte)
    rows = []
    for v in vars_main:
        ref_vals = df.loc[v, ref_cols]
        t = pd.to_numeric(df.loc[v, target_col], errors="coerce")
        if not np.isfinite(t):
            continue
        p = _pct_leq(ref_vals, float(t))  # Anteil refs <= target
        if not np.isfinite(p):
            continue
        rows.append((v, float(t), float(p)))

    dep = pd.DataFrame(rows, columns=["var", "target", "pct"]).set_index("var")

    # ---------------------------
    # 1) Trendtext (präzise wording)
    # ---------------------------
    n = len(dep)
    if n == 0:
        trend_text = "Es liegen keine auswertbaren Wettervariablen zur Trendanalyse vor."
        top_text = ""
    else:
        n_higher = int((dep["pct"] > 50).sum())
        n_lower = int((dep["pct"] < 50).sum())
        n_equal = n - n_higher - n_lower

        share_higher = 100.0 * n_higher / n
        share_lower = 100.0 * n_lower / n

        # Hard clamp (nur zur Sicherheit gegen Rundungs-/Edgecases)
        share_higher = float(np.clip(share_higher, 0.0, 100.0))
        share_lower = float(np.clip(share_lower, 0.0, 100.0))

        # Hilfstext abhängig von Anzahl Variablen
        if n == 1:
            # genau eine Wettervariable: NICHT "alle Wettervariablen"
            single_var = dep.index[0]
            single_label = labels.get(single_var, single_var)
            direction = "höhere" if dep.loc[single_var, "pct"] > 50 else "geringere" if dep.loc[single_var, "pct"] < 50 else "ähnliche"
            trend_text = (
                f"Die betrachtete Wettervariable ({single_label}) zeigt beim Zielprofil "
                f"eine {direction} Abhängigkeit als die Referenzlastprofile."
            )
        else:
            # mehrere Variablen: "alle betrachteten" oder Prozentangabe
            if share_higher >= trend_threshold:
                if share_higher >= 99.5:
                    trend_text = (
                        "Alle betrachteten Wettervariablen zeigen beim Zielprofil "
                        "eine höhere Abhängigkeit als die Referenzlastprofile."
                    )
                else:
                    trend_text = (
                        f"{share_higher:.0f}% der betrachteten Wettervariablen zeigen beim Zielprofil "
                        "eine höhere Abhängigkeit als die Referenzlastprofile."
                    )
            elif share_lower >= trend_threshold:
                if share_lower >= 99.5:
                    trend_text = (
                        "Alle betrachteten Wettervariablen zeigen beim Zielprofil "
                        "eine geringere Abhängigkeit als die Referenzlastprofile."
                    )
                else:
                    trend_text = (
                        f"{share_lower:.0f}% der betrachteten Wettervariablen zeigen beim Zielprofil "
                        "eine geringere Abhängigkeit als die Referenzlastprofile."
                    )
            else:
                # gemischt
                if n_equal > 0:
                    trend_text = (
                        "Die Wetterabhängigkeit des Zielprofils ist variablenabhängig: "
                        f"{share_higher:.0f}% der Variablen zeigen eine höhere, "
                        f"{share_lower:.0f}% eine geringere und "
                        f"{100.0 * n_equal / n:.0f}% eine ähnliche Abhängigkeit "
                        "im Vergleich zu den Referenzen."
                    )
                else:
                    trend_text = (
                        "Die Wetterabhängigkeit des Zielprofils ist variablenabhängig: "
                        f"{share_higher:.0f}% der Variablen zeigen eine höhere und "
                        f"{share_lower:.0f}% eine geringere Abhängigkeit "
                        "im Vergleich zu den Referenzen."
                    )

        # ---------------------------
        # 2) Zwei stärkste Abweichungen
        # ---------------------------
        dep["dist50"] = (dep["pct"] - 50).abs()
        top2 = dep.sort_values("dist50", ascending=False).head(2)

        def fmt_line(v, t, p):
            name = labels.get(v, v)
            k = _klasse_typisch(p, typisch_band=typisch_band, eher_band=eher_band)
            return f"- {name}: ρ = {t:.2f} ({k} im Vergleich zu den Referenzen)"

        top_lines = [
            fmt_line(v, float(row["target"]), float(row["pct"]))
            for v, row in top2.iterrows()
        ]
        top_text = "Die stärksten Zusammenhänge zeigen:\n" + "\n".join(top_lines)

    # ---------------------------
    # 3) Overall-Satz + Brücke
    # ---------------------------
    overall_text = ""
    bridge_text = ""

    overall_pct = np.nan
    overall_target = np.nan

    if overall_name is not None and overall_name in df.index:
        ref_vals = df.loc[overall_name, ref_cols]
        t = pd.to_numeric(df.loc[overall_name, target_col], errors="coerce")
        if np.isfinite(t):
            overall_target = float(t)
            p = _pct_leq(ref_vals, overall_target)
            if np.isfinite(p):
                overall_pct = float(p)
                k = _klasse_typisch(overall_pct, typisch_band=typisch_band, eher_band=eher_band)
                overall_label = labels.get(overall_name, overall_name)
                overall_text = (
                    f"{overall_label}: Perzentil {overall_pct:.1f} "
                    f"({k} im Vergleich zu den Referenzen)."
                )

    # Brückensatz, wenn (a) Trend sagt "höher" (b) Overall ist sehr niedrig
    # -> erklärt, dass einzelne Variable hoch sein kann, Gesamtmaß aber niedrig.
    if n > 0 and np.isfinite(overall_pct):
        trend_is_higher = False
        if n == 1:
            # bei einer Variable: "höher" wenn pct > 50
            only_var = dep.index[0]
            trend_is_higher = float(dep.loc[only_var, "pct"]) > 50
        else:
            trend_is_higher = share_higher >= trend_threshold

        if trend_is_higher and overall_pct <= 25:
            bridge_text = (
                "Hinweis: Obwohl einzelne Wettervariablen beim Zielprofil stärker korrelieren, "
                "ist die Gesamtwetterabhängigkeit niedrig – das deutet darauf hin, dass sich die Last "
                "insgesamt schlechter durch Wetter allein erklären lässt."
            )
        elif (not trend_is_higher) and overall_pct >= 75:
            bridge_text = (
                "Hinweis: Obwohl einzelne Wettervariablen beim Zielprofil schwächer ausfallen, "
                "ist die Gesamtwetterabhängigkeit hoch – das deutet darauf hin, dass das Wettermodell "
                "für diese Last insgesamt gut erklärt."
            )

    # Zusammenbauen
    parts = [trend_text]
    if top_text:
        parts.append(top_text)
    if overall_text:
        parts.append(overall_text)
    if bridge_text:
        parts.append(bridge_text)

    return "\n\n".join(parts)


##################################################################################


def weather_corr(df: pd.DataFrame, load_id: str, weather_vars: List[str], test_months: List[int] = [1, 4, 7, 11]) -> float:
    """
    Gesamt-Wetterabhängigkeit measured as Spearman correlation between
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
    """
    df = all_corr.copy()

    ref_cols = [c for c in df.columns if c != target_col]
    vars_order = list(df.index)

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

    for i, var in enumerate(vars_order):
        vals = df.loc[var, ref_cols].dropna().values.astype(float)

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
                fillcolor="rgba(220,220,220,0.20)",
                opacity=1.0,
                spanmode="hard",
                width=0.8,
            )
        )

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

    # Target points only (no line)
    target_vals = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(vars_order)),
            y=target_vals,
            mode="markers",
            name=f"Target {target_column}",
            marker=dict(symbol="diamond", size=10, color=c_target),
            hovertemplate="Target: %{y:.3f}<extra></extra>",
            showlegend=True,
        )
    )

    x_labels = [WEATHER_LABELS.get(v, v) for v in vars_order]

    # --- dynamic y range (robust) ---
    vals_ref = df[ref_cols].to_numpy(dtype=float).ravel() if ref_cols else np.array([], dtype=float)
    vals_tgt = target_vals.ravel()
    vals = np.concatenate([vals_ref, vals_tgt])
    vals = vals[np.isfinite(vals)]

    if vals.size:
        q_low, q_high = np.quantile(vals, [0.05, 0.95])
        span = max(q_high - q_low, 1e-6)
        pad = 0.15 * span
        lo = q_low - pad
        hi = q_high + pad
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
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


def generate_distribution_comparison(data_df: pd.DataFrame, benchmark_columns: List[str], target_column: str):
    """
    Entry point used by the dashboard.

    Returns
    -------
    (fig, capture)
    """
    ref_cols = [f"load_{x}" for x in benchmark_columns]
    target_col = f"load_{target_column}"

    weather_vars = ["temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]

    weather_cols=[x + f"_{target_column}" for x in weather_vars]
    missing_stats = pd.DataFrame({
        "n_missing": data_df[weather_cols].isna().sum(),
        "pct_missing": data_df[weather_cols].isna().mean() * 100
    })
    missing_stats = missing_stats[missing_stats.pct_missing > 30]

    # Drop weather vars with high proportion of missing values (> 30%)
    high_missing = missing_stats.index

    base_vars_high_missing = high_missing.to_series().astype(str).str.rsplit("_", n=1).str[0]
    weather_vars_to_drop = sorted(set(base_vars_high_missing).intersection(weather_vars))
    weather_vars = [v for v in weather_vars if v not in weather_vars_to_drop]

    all_cols = ref_cols + [target_col]
    all_col_names = benchmark_columns + [target_column]

    all_corr_series = []
    for i in range(len(all_cols)):
        load_name = all_cols[i]
        load_id = all_col_names[i]
        weather_feature_cols = [f"{v}_{load_id}" for v in weather_vars]

        corr = data_df[[load_name] + weather_feature_cols].corr(method="spearman")
        corr = corr[load_name].drop(load_name)
        corr.index = weather_vars
        all_corr_series.append(corr.rename(load_name))

    all_corr = pd.concat(all_corr_series, axis=1)
    all_corr = all_corr.dropna(how="all")

    all_overall = []
    for col_id in all_col_names:
        all_overall.append(weather_corr(data_df, col_id, weather_vars=list(all_corr.index)))

    all_corr.loc["Gesamt-Wetterabhängigkeit", :] = all_overall

    fig = plot_dependency_violinplot(
        all_corr,
        target_col=target_col,
        target_column=target_column,
        title="Wetterabhängigkeit der Last",
    )

    capture = describe_weather_dependency_de(
        all_corr,
        target_col,
        weather_vars=list(all_corr.index),
        overall_name="Gesamt-Wetterabhängigkeit",
    )

    return fig, capture
