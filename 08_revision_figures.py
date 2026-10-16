"""CropWhisperAI - figures added in revision 1 (run after 07_revision_analyses.py).
fig12_examples.png      successful warning, missed episode and false alarm of the on-device model (30 min)
fig13_lead_times.png    distribution of lead times over all warned episodes and alarm persistence
All figures use one font family (Liberation Sans, metric-compatible with Arial) at 8-9 pt.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cw_common import *

FONT = "Liberation Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#4a3aa7", red="#e34948", gray="#8a8984", ink="#0b0b0b")
R = "results/revision"
d = load("data.csv"); s = raw_thermal(d); eps = episodes(s)
meta = json.load(open("esp32/model_meta.json")); THR = meta["logit_threshold"]


def sig(x): return 1 / (1 + np.exp(-x))


def day_panel(ax0, ax1, day0, title, oof, p8, hours=(6, 19)):
    w = (d.Timestamp >= day0 + pd.Timedelta(hours=hours[0])) & (d.Timestamp < day0 + pd.Timedelta(hours=hours[1]))
    dd = d[w]
    ax0.plot(dd.Timestamp, dd[AIR_T], color=C["orange"], lw=1.4, label="Air temperature")
    ax0.plot(dd.Timestamp, dd[SOIL_T], color=C["blue"], lw=1.4, label="Soil temperature")
    ax0.axhline(AIR_THR, color=C["orange"], ls="--", lw=0.8); ax0.axhline(SOIL_THR, color=C["blue"], ls="--", lw=0.8)
    for a, b in eps:
        if dd.Timestamp.min() <= d.Timestamp[a] <= dd.Timestamp.max():
            ax0.axvspan(d.Timestamp[a], d.Timestamp[min(b, dd.index.max())], color=C["red"], alpha=0.10, lw=0)
            for x in (ax0, ax1): x.axvline(d.Timestamp[a], color=C["red"], lw=1.0)
    ax0.set_ylabel("Temperature (°C)"); ax0.set_title(title, loc="left")
    q = p8[(p8.Timestamp >= dd.Timestamp.min()) & (p8.Timestamp <= dd.Timestamp.max())]
    q = q.set_index("Timestamp").reindex(dd.Timestamp).reset_index()          # unscored steps -> gaps in the line
    q["alarm"] = q.alarm.fillna(0)
    o = oof[(oof.Timestamp >= dd.Timestamp.min()) & (oof.Timestamp <= dd.Timestamp.max())]
    ax1.plot(q.Timestamp, sig(q.score), color=C["violet"], lw=1.4, label="On-device LR risk")
    la = q[q.alarm.astype(bool)]; ta = o[o["Rule: 30-min trend extrapolation|alarm"] == 1]
    ax1.plot(la.Timestamp, [1.10] * len(la), "|", color=C["violet"], ms=7, mew=1.5, label="On-device LR alarm")
    ax1.plot(ta.Timestamp, [1.25] * len(ta), "|", color=C["gray"], ms=7, mew=1.5, label="Trend-rule alarm")
    ax1.set_ylim(-0.03, 1.35); ax1.set_ylabel("Risk (probability)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def fig_examples():
    oof = pd.read_csv("results/oof_H30.csv", parse_dates=["Timestamp"])
    p8 = pd.read_csv(f"{R}/oof_LR8_H30.csv"); p8["Timestamp"] = d.Timestamp.iloc[p8.idx].values
    E = pd.read_csv(f"{R}/episodes_H30.csv", parse_dates=["onset_time"]); E = E[E.model == "On-device LR"]
    runs = pd.read_csv(f"{R}/alarm_runs_H30.csv", parse_dates=["start", "end"]); runs = runs[runs.model == "On-device LR"]
    ok = E[(E.warned == 1) & (E.onset_time.dt.month == 7)].iloc[0].onset_time.normalize()
    miss = E[E.warned == 0].onset_time
    f0 = runs[runs.true_alarm == 0].copy()
    on_days = {pd.Timestamp(d.Timestamp[a]).normalize() for a, b in eps}
    f0 = f0[~f0.start.dt.normalize().isin(on_days)]                            # a day without any episode
    fa = f0.sort_values(["length_min", "Tair"], ascending=False).start
    panels = [(ok, "(a) Successful warning")]
    if len(miss): panels.append((miss.iloc[0].normalize(), "(b) Missed episode"))
    if len(fa): panels.append((fa.iloc[0].normalize(), f"({'c' if len(miss) else 'b'}) False alarm"))
    fig, ax = plt.subplots(2, len(panels), figsize=(7.2, 4.0), sharey="row", sharex="col", gridspec_kw=dict(height_ratios=[1.2, 1]))
    ax = np.atleast_2d(ax).reshape(2, len(panels))
    for j, (dy, t) in enumerate(panels):
        day_panel(ax[0, j], ax[1, j], dy, f"{t}, {dy:%d %b}", oof, p8)
        if j: ax[0, j].set_ylabel(""); ax[1, j].set_ylabel("")
        for lab in ax[1, j].get_xticklabels(): lab.set_rotation(0)
        ax[1, j].xaxis.set_major_locator(mdates.HourLocator(byhour=[8, 12, 16]))
        ax[0, j].xaxis.set_major_locator(mdates.HourLocator(byhour=[8, 12, 16])); ax[0, j].tick_params(labelbottom=False)
    h0, l0 = ax[0, 0].get_legend_handles_labels(); h1, l1 = ax[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("figures/fig12_examples.png"); plt.close(fig)
    return [(str(p[0].date()), p[1]) for p in panels]


def fig_leads():
    E = pd.read_csv(f"{R}/episodes_H30.csv")
    order = ["Trend rule", "Logistic regression", "LightGBM", "On-device LR"]
    col = {"Trend rule": C["gray"], "Logistic regression": C["orange"], "LightGBM": C["blue"], "On-device LR": C["violet"]}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    bins = np.arange(2.5, 36, 5)
    for i, m in enumerate(order):
        g = E[(E.model == m) & (E.warned == 1)]
        cnt, _ = np.histogram(g.lead_min, bins)
        ax[0].bar(np.arange(5, 35, 5) + (i - 1.5) * 1.1, cnt, width=1.05, color=col[m], label=f"{m} ({len(g)}/42)")
    ax[0].set_xlabel("Lead time (min)"); ax[0].set_ylabel("Warned episodes"); ax[0].set_xticks(range(5, 35, 5))
    ax[0].set_title("(a) Lead time, 30-min horizon", loc="left"); ax[0].legend(fontsize=7, loc="upper left")
    frac = [E[(E.model == m) & (E.warned == 1)].continuous_to_onset.mean() for m in order]
    miss = [int((E[E.model == m].warned == 0).sum()) for m in order]
    y = np.arange(len(order))
    ax[1].barh(y, frac, color=[col[m] for m in order])
    for yy, f, mi in zip(y, frac, miss): ax[1].text(f + 0.01, yy, f"{f:.0%}  (missed {mi})", va="center", fontsize=8)
    ax[1].set_yticks(y); ax[1].set_yticklabels(order); ax[1].set_xlim(0, 1.25); ax[1].set_xlabel("Share of warned episodes")
    ax[1].set_title("(b) Alarm stays on until onset", loc="left")
    fig.tight_layout(); fig.savefig("figures/fig13_lead_times.png"); plt.close(fig)


if __name__ == "__main__":
    print(fig_examples()); fig_leads(); print("ok")
