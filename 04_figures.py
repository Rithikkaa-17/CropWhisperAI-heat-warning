"""CropWhisperAI - publication figures (300 dpi PNG) from results/*.  Run after scripts 01-03."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from cw_common import *

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300",
         violet="#4a3aa7", red="#e34948", gray="#8a8984", ink="#0b0b0b", muted="#52514e")
MODEL_C = {"LightGBM": C["blue"], "Logistic regression": C["orange"], "Random forest": C["aqua"], "MLP (64-32)": C["yellow"],
           "Rule: 30-min trend extrapolation": C["violet"], "Rule: threshold proximity": C["magenta"], "Time-of-day only (LR)": C["gray"]}
SHORTN = {"Rule: 30-min trend extrapolation": "Trend rule", "Rule: threshold proximity": "Proximity rule",
          "Time-of-day only (LR)": "Time of day only", "MLP (64-32)": "MLP", "Logistic regression": "Logistic regression",
          "Random forest": "Random forest", "LightGBM": "LightGBM"}
d = load("data.csv")
s = raw_thermal(d); eps = episodes(s)


def fig_dataset():
    g = d.set_index("Timestamp").resample("D")
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    mx = g[[AIR_T, SOIL_T]].max()
    ax[0].plot(mx.index, mx[AIR_T], color=C["orange"], lw=1.6, label="Air temperature (daily max)")
    ax[0].plot(mx.index, mx[SOIL_T], color=C["blue"], lw=1.6, label="Soil temperature (daily max)")
    ax[0].axhline(AIR_THR, color=C["orange"], ls="--", lw=0.9, label="Air rule (35 °C)"); ax[0].axhline(SOIL_THR, color=C["blue"], ls="--", lw=0.9, label="Soil rule (32 °C)")
    days = sorted({d.Timestamp[a].normalize() for a, b in eps})
    ax[0].plot(days, [19.5] * len(days), "|", color=C["red"], ms=8, mew=1.6, label="Heat-stress episode day")
    ax[0].set_ylabel("°C"); ax[0].set_ylim(18, 42); ax[0].legend(loc="lower left", ncol=3, fontsize=8, bbox_to_anchor=(0.0, 0.07))
    ax[0].set_title("(a) Daily maximum temperatures and heat-stress episode days", loc="left", fontsize=9)
    mn = g[[SOIL_M, WL]].mean()
    ax[1].plot(mn.index, mn[SOIL_M], color=C["aqua"], lw=1.6, label="Soil moisture (daily mean)")
    ax[1].plot(mn.index, mn[WL], color=C["violet"], lw=1.6, label="Reservoir level (daily mean)")
    ax[1].axhline(30, color=C["aqua"], ls="--", lw=0.9); ax[1].axhline(25, color=C["violet"], ls="--", lw=0.9)
    ax[1].set_ylabel("%"); ax[1].set_ylim(0, 100); ax[1].legend(loc="lower left", fontsize=8, ncol=2)
    ax[1].set_title("(b) Water status (dashed: water-deficit rules, never reached)", loc="left", fontsize=9)
    import matplotlib.dates as mdates
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig("figures/fig3_dataset_overview.png"); plt.close(fig)


def fig_diurnal():
    h = d.Timestamp.dt.hour
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for col, c, lab in ((AIR_T, C["orange"], "Air"), (SOIL_T, C["blue"], "Soil")):
        q = d.groupby(h)[col].quantile([0.25, 0.5, 0.75]).unstack()
        ax[0].fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.18, lw=0)
        ax[0].plot(q.index, q[0.5], color=c, lw=1.8, label=f"{lab} (median, IQR)")
    ax[0].set_xlabel("Hour of day (IST)"); ax[0].set_ylabel("°C"); ax[0].set_xticks(range(0, 24, 3)); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) Diurnal temperature cycle", loc="left", fontsize=9)
    oh = pd.Series([d.Timestamp[a].hour for a, b in eps]).value_counts().reindex(range(24), fill_value=0)
    ax[1].bar(oh.index, oh.values, color=C["red"], width=0.8)
    ax[1].set_xlabel("Hour of day (IST)"); ax[1].set_ylabel("Episode onsets"); ax[1].set_xticks(range(0, 24, 3))
    ax[1].set_title(f"(b) Onset time of the {len(eps)} heat-stress episodes", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig4_diurnal.png"); plt.close(fig)


def fig_pr():
    o = pd.read_csv("results/oof_H30.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    m = pd.read_csv("results/metrics_cv.csv"); m = m[m.horizon_min == 30].set_index("model")
    for name in ["LightGBM", "Logistic regression", "Rule: 30-min trend extrapolation", "Rule: threshold proximity"]:
        p, r, _ = precision_recall_curve(o.y, o[name + "|score"])
        ax.plot(r, p, color=MODEL_C[name], lw=1.8, label=f"{SHORTN[name]} (AP {m.loc[name,'AUPRC']:.3f})")
    ax.axhline(o.y.mean(), color=C["gray"], ls=":", lw=1); ax.text(0.62, o.y.mean() + 0.02, "prevalence", color=C["muted"], fontsize=8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.47)); ax.set_title("30-min-ahead warning, blocked CV", loc="left", fontsize=9)
    fig.savefig("figures/fig5_pr_curves.png"); plt.close(fig)


def fig_horizon():
    m = pd.read_csv("results/metrics_cv.csv")
    names = ["LightGBM", "Logistic regression", "Random forest", "Rule: 30-min trend extrapolation"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    w = 0.2
    for i, n in enumerate(names):
        mm = m[m.model == n].sort_values("horizon_min"); x = np.arange(3) + (i - 1.5) * w
        ax[0].bar(x, mm.AUPRC, w * 0.9, color=MODEL_C[n], label=SHORTN[n])
        ax[0].errorbar(x, mm.AUPRC, yerr=[mm.AUPRC - mm.AUPRC_lo, mm.AUPRC_hi - mm.AUPRC], fmt="none", ecolor=C["ink"], lw=0.8, capsize=2)
        ax[1].bar(x, mm.false_alarms_per_day, w * 0.9, color=MODEL_C[n])
    for a in ax: a.set_xticks(range(3)); a.set_xticklabels(["15 min", "30 min", "60 min"]); a.set_xlabel("Warning horizon")
    ax[0].set_ylabel("AUPRC (95% CI)"); ax[0].set_ylim(0.5, 1.0); ax[0].set_title("(a) Ranking quality", loc="left", fontsize=9)
    ax[1].set_ylabel("False alarms per day"); ax[1].set_title("(b) False alarms at the operating threshold", loc="left", fontsize=9)
    fig.legend(*ax[0].get_legend_handles_labels(), fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); fig.savefig("figures/fig6_horizons.png"); plt.close(fig)


def fig_ablation():
    import os
    if os.path.exists("results/revision/ablation_ci.csv"):        # revision: with 95% day-block bootstrap CIs
        a = pd.read_csv("results/revision/ablation_ci.csv").sort_values("AUPRC")
        t = pd.read_csv("results/ablation.csv"); t = t[(t.horizon_min == 30) & (t.subset == "No sensors (time of day)")]
        a = pd.concat([t.assign(AUPRC_lo=np.nan, AUPRC_hi=np.nan), a]).reset_index(drop=True)
    else:
        a = pd.read_csv("results/ablation.csv"); a = a[a.horizon_min == 30]
        a = a[a.subset != "All minus soil moisture & water level"].sort_values("AUPRC")
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    cols = [C["blue"] if s != "No sensors (time of day)" else C["gray"] for s in a.subset]
    ax.barh(a.subset, a.AUPRC, color=cols, height=0.6)
    if "AUPRC_lo" in a:
        m = a.AUPRC_lo.notna()
        ax.errorbar(a.AUPRC[m], np.arange(len(a))[m.values], xerr=[a.AUPRC[m] - a.AUPRC_lo[m], a.AUPRC_hi[m] - a.AUPRC[m]],
                    fmt="none", ecolor=C["ink"], elinewidth=0.9, capsize=2.5)
    for y_, (v, hi) in enumerate(zip(a.AUPRC, a.get("AUPRC_hi", a.AUPRC))):
        x = (hi if pd.notna(hi) else v) + 0.015
        ax.text(x, y_, f"{v:.3f}", va="center", fontsize=8, color=C["ink"])
    ax.set_xlim(0, 1.12); ax.set_xlabel("AUPRC, 30-min horizon (LightGBM, blocked CV)"); ax.grid(axis="y", visible=False)
    fig.savefig("figures/fig7_sensor_ablation.png"); plt.close(fig)


def fig_shap():
    imp = pd.read_csv("results/shap_importance.csv").head(15).iloc[::-1]
    gc = {"DHT22": C["orange"], "DS18B20": C["blue"], "BH1750": C["yellow"], "Capacitive": C["aqua"], "WaterLevel": C["violet"],
          "Time of day": C["gray"], "DHT22 x DS18B20 (gradient)": C["magenta"]}
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.barh(imp.feature, imp.mean_abs_shap, color=[gc.get(g, C["gray"]) for g in imp.group], height=0.65)
    from matplotlib.patches import Patch
    present = [g for g in gc if g in set(imp.group)]
    ax.legend(handles=[Patch(color=gc[g], label=g) for g in present], fontsize=8, loc="lower right")
    ax.set_xlabel("Mean |SHAP value| (log-odds)"); ax.grid(axis="y", visible=False)
    ax.set_title("Top 15 features, 30-min LightGBM", loc="left", fontsize=9)
    fig.savefig("figures/fig8_shap.png"); plt.close(fig)


def fig_tiny():
    t = pd.read_csv("results/tinyml_candidates.csv")
    kc = {"lgb": C["blue"], "lr": C["orange"], "mlp": C["aqua"]}
    kl = {"lgb": "LightGBM (compact)", "lr": "Logistic regression", "mlp": "MLP 16-8"}
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for k in ("lgb", "lr", "mlp"):
        tt = t[t.kind == k]
        ax.scatter(tt.model_bytes, tt.AUPRC, s=[36 if n == 8 else 64 for n in tt.n_features], color=kc[k], label=kl[k],
                   edgecolor="white", linewidth=1.5, zorder=3)
    v = json.load(open("results/tinyml_verify.json"))
    sel = t[t.model == v["chosen"]].iloc[0]
    ax.annotate("deployed", (sel.model_bytes, sel.AUPRC), xytext=(10, -18), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    ax.set_xscale("log"); ax.set_xlabel("Model parameter memory (bytes, log scale)"); ax.set_ylabel("AUPRC (30 min)")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("On-device candidates (small = 8, large = 12 features)", loc="left", fontsize=9)
    fig.savefig("figures/fig9_tinyml_pareto.png"); plt.close(fig)


def fig_example():
    o = pd.read_csv("results/oof_H30.csv", parse_dates=["Timestamp"])
    # choose the first July episode day for illustration
    a0 = [a for a, b in eps if d.Timestamp[a].month == 7][0]
    day0 = d.Timestamp[a0].normalize()
    w = (d.Timestamp >= day0 + pd.Timedelta(hours=6)) & (d.Timestamp < day0 + pd.Timedelta(hours=19))
    dd = d[w]; oo = o[(o.Timestamp >= dd.Timestamp.min()) & (o.Timestamp <= dd.Timestamp.max())]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax[0].plot(dd.Timestamp, dd[AIR_T], color=C["orange"], lw=1.6, label="Air temperature")
    ax[0].plot(dd.Timestamp, dd[SOIL_T], color=C["blue"], lw=1.6, label="Soil temperature")
    ax[0].axhline(AIR_THR, color=C["orange"], ls="--", lw=0.8); ax[0].axhline(SOIL_THR, color=C["blue"], ls="--", lw=0.8)
    for a, b in eps:
        if dd.Timestamp.min() <= d.Timestamp[a] <= dd.Timestamp.max():
            ax[0].axvspan(d.Timestamp[a], d.Timestamp[min(b, dd.index.max())], color=C["red"], alpha=0.10, lw=0)
            for x in ax: x.axvline(d.Timestamp[a], color=C["red"], lw=1.0)
    ax[0].set_ylabel("°C"); ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_title(f"(a) {day0:%d %B %Y}: shaded = heat-stress episode, red line = onset", loc="left", fontsize=9)
    ax[1].plot(oo.Timestamp, oo["LightGBM|score"], color=C["blue"], lw=1.6, label="LightGBM risk (30 min ahead)")
    tr = oo["Rule: 30-min trend extrapolation|score"]
    la = oo[oo["LightGBM|alarm"] == 1]; ra = oo[oo["Rule: 30-min trend extrapolation|alarm"] == 1]
    ax[1].plot(la.Timestamp, [1.08] * len(la), "|", color=C["blue"], ms=7, mew=1.5, label="LightGBM alarm")
    ax[1].plot(ra.Timestamp, [1.2] * len(ra), "|", color=C["violet"], ms=7, mew=1.5, label="Trend-rule alarm")
    ax[1].set_ylim(-0.02, 1.3); ax[1].set_ylabel("Probability"); ax[1].legend(fontsize=8, loc="center left")
    ax[1].set_title("(b) Out-of-fold risk score and alarms (points inside an episode are not scored)", loc="left", fontsize=9)
    import matplotlib.dates as mdates
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout(); fig.savefig("figures/fig10_example_day.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_dataset, fig_diurnal, fig_pr, fig_horizon, fig_ablation, fig_shap, fig_tiny, fig_example):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
