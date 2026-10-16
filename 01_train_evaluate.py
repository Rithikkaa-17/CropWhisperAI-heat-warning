"""CropWhisperAI - heat-stress early warning: blocked cross-validation of rules vs ML models.

Outputs (results/):
  metrics_cv.csv          pooled out-of-fold metrics per horizon x model (+ 95% day-block bootstrap CIs)
  oof_H{h}.csv            out-of-fold scores/alarms per eligible sample
  metrics_chrono.csv      strict chronological hold-out (train Jun 1-Jul 15, test Jul 16-Sep 15)
  paired_tests.csv        paired day-block bootstrap: each ML model vs best rule baseline
Run:  python 01_train_evaluate.py
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
from cw_common import *

warnings.filterwarnings("ignore")
SEED = 42
HORIZONS = [3, 6, 12]           # 15, 30, 60 min
N_BOOT = 1000
os.makedirs("results", exist_ok=True)


# ---------------- models ----------------
class RuleProximity:
    """Score = how close the nearer sensor is to its threshold (no learning)."""
    name = "Rule: threshold proximity"
    def fit(self, X, y): return self
    def score(self, X): return -np.minimum(X["Tair_margin"].values, X["Tsoil_margin"].values)

class RuleTrend:
    """Linear extrapolation of the last 30-min slope to the forecast horizon."""
    name = "Rule: 30-min trend extrapolation"
    def __init__(self, h): self.h = h
    def fit(self, X, y): return self
    def score(self, X):
        pa = X["Tair"] + X["Tair_d30"] / 6 * self.h - AIR_THR
        ps = X["Tsoil"] + X["Tsoil_d30"] / 6 * self.h - SOIL_THR
        return np.maximum(pa, ps).fillna(-99).values

class SK:
    def __init__(self, name, est, cols=None):
        self.name, self.est, self.cols = name, est, cols
    def fit(self, X, y):
        Xc = X if self.cols is None else X[self.cols]
        self.est.fit(Xc.fillna(0).values if not isinstance(self.est, lgb.LGBMClassifier) else Xc, y)
        return self
    def score(self, X):
        Xc = X if self.cols is None else X[self.cols]
        return self.est.predict_proba(Xc.fillna(0).values if not isinstance(self.est, lgb.LGBMClassifier) else Xc)[:, 1]


def make_models(h):
    pos_w = 20
    return [
        RuleProximity(),
        RuleTrend(h),
        SK("Time-of-day only (LR)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), ["hour_sin", "hour_cos"]),
        SK("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))),
        SK("Random forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                   class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        SK("LightGBM", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                          scale_pos_weight=pos_w, random_state=SEED, verbose=-1)),
        SK("MLP (64-32)", make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, early_stopping=True,
                                                                        max_iter=300, random_state=SEED))),
    ]


# ---------------- evaluation helpers ----------------
def best_f1_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 300)))
    best = (0, qs[-1])
    for t in qs:
        p = s >= t
        tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = ((~p) & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]: best = (f1, t)
    return best[1]


def event_metrics(idx, alarm, onsets, h, n_days):
    """idx: dataframe index of scored samples; alarm: bool per scored sample."""
    a_set = set(idx[alarm])
    detected, leads = 0, []
    for o in onsets:
        win = [t for t in range(o - h, o) if t in a_set]
        if win:
            detected += 1; leads.append((o - min(win)) * STEP_MIN)
    # false-alarm episodes: alarm runs (consecutive indices) with no onset within H after run start
    fa, run_start, prev = 0, None, None
    on_arr = np.array(onsets)
    for t in sorted(a_set):
        if prev is None or t != prev + 1:
            if run_start is not None and not ((on_arr > run_start) & (on_arr <= prev + h)).any():
                fa += 1
            run_start = t
        prev = t
    if run_start is not None and not ((on_arr > run_start) & (on_arr <= prev + h)).any():
        fa += 1
    return dict(events=len(onsets), detected=detected,
                event_recall=detected / max(len(onsets), 1),
                median_lead_min=float(np.median(leads)) if leads else np.nan,
                false_alarms_per_day=fa / n_days)


def sample_metrics(y, s, alarm):
    tp = (alarm & (y == 1)).sum(); fp = (alarm & (y == 0)).sum(); fn = ((~alarm) & (y == 1)).sum()
    out = dict(AUROC=roc_auc_score(y, s), AUPRC=average_precision_score(y, s),
               precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), F1=2 * tp / max(2 * tp + fp + fn, 1),
               prevalence=y.mean())
    return out


def run_split(models_fn, X, y, elig, train_mask, test_mask, h, days_series):
    """Fit on train (threshold from inner chronological 80/20 split), score test."""
    res = {}
    tr = np.where(train_mask & elig)[0]
    te = np.where(test_mask & elig)[0]
    tr_days = np.array(sorted(days_series.iloc[tr].unique()))
    cut = tr_days[int(len(tr_days) * 0.8)]
    inner_tr = tr[days_series.iloc[tr].values < cut]
    inner_va = tr[days_series.iloc[tr].values >= cut]
    # guarantee positives in inner validation; otherwise use random-day 20%
    if y[inner_va].sum() < 5:
        rng = np.random.RandomState(SEED)
        vd = rng.choice(tr_days, max(1, len(tr_days) // 5), replace=False)
        m = days_series.iloc[tr].isin(vd).values
        inner_tr, inner_va = tr[~m], tr[m]
    for m in models_fn(h):
        m.fit(X.iloc[inner_tr], y[inner_tr])
        thr = best_f1_threshold(y[inner_va], m.score(X.iloc[inner_va]))
        m.fit(X.iloc[tr], y[tr])
        res[m.name] = (te, m.score(X.iloc[te]), thr)
    return res


def boot_ci(df, fn, n=N_BOOT, seed=SEED):
    """Day-block bootstrap of a metric function over a per-sample dataframe with a 'day' column."""
    rng = np.random.RandomState(seed)
    days = df["day"].unique()
    groups = {k: v.index.values for k, v in df.groupby("day")}
    vals = []
    for _ in range(n):
        pick = rng.choice(days, len(days), replace=True)
        ix = np.concatenate([groups[k] for k in pick])
        sub = df.loc[ix]
        if sub.y.sum() < 3: continue
        vals.append(fn(sub))
    return np.percentile(vals, [2.5, 97.5]), np.array(vals)


def main():
    t0 = time.time()
    d = load("data.csv")
    X = build_features(d)
    day = d.Timestamp.dt.normalize()
    fold, chunks = day_blocks(d, 10)
    n_days_total = day.nunique()
    rows, paired_rows = [], []
    for h in HORIZONS:
        y, elig, eps = make_labels(d, h)
        onsets_all = np.array([a for a, b in eps])
        pooled = {}
        for k in range(10):
            test_days = set(chunks[k])
            # embargo: drop training days adjacent (+-1 day) to the test block
            emb = set()
            for td in test_days:
                emb.update({td - np.timedelta64(1, "D"), td + np.timedelta64(1, "D")})
            test_mask = (fold == k)
            train_mask = (~test_mask) & (~day.isin(emb).values)
            out = run_split(make_models, X, y, elig, train_mask, test_mask, h, day)
            for name, (te, s, thr) in out.items():
                pooled.setdefault(name, []).append(pd.DataFrame({"idx": te, "score": s, "alarm": s >= thr, "fold": k}))
        oof = None
        for name, parts in pooled.items():
            p = pd.concat(parts).sort_values("idx").reset_index(drop=True)
            p["y"] = y[p.idx]; p["day"] = day.iloc[p.idx].values
            sm = sample_metrics(p.y.values, p.score.values, p.alarm.values)
            em = event_metrics(p.idx.values, p.alarm.values, onsets_all, h, n_days_total)
            (lo, hi), _ = boot_ci(p, lambda s: average_precision_score(s.y, s.score))
            rows.append(dict(horizon_min=h * 5, model=name, **sm, AUPRC_lo=lo, AUPRC_hi=hi, **em))
            col = name
            if oof is None:
                oof = p[["idx", "y", "day", "fold"]].copy()
                oof["Timestamp"] = d.Timestamp.iloc[p.idx].values
            oof[col + "|score"] = p.score.values
            oof[col + "|alarm"] = p.alarm.values.astype(int)
            print(f"H={h*5:>2}min  {name:<35} AUPRC={sm['AUPRC']:.3f} [{lo:.3f},{hi:.3f}] AUROC={sm['AUROC']:.3f} "
                  f"F1={sm['F1']:.3f} evRecall={em['event_recall']:.2f} lead={em['median_lead_min']} FA/day={em['false_alarms_per_day']:.2f}", flush=True)
        oof.to_csv(f"results/oof_H{h*5}.csv", index=False)
        # paired bootstrap: each learned model vs best rule
        rules = [n for n in pooled if n.startswith("Rule")]
        best_rule = max(rules, key=lambda n: average_precision_score(oof.y, oof[n + "|score"]))
        rng = np.random.RandomState(SEED)
        dgroups = {k: v.index.values for k, v in oof.groupby("day")}
        udays = oof.day.unique()
        for name in pooled:
            if name.startswith("Rule"): continue
            diffs = []
            for _ in range(N_BOOT):
                ix = np.concatenate([dgroups[k] for k in rng.choice(udays, len(udays))])
                sub = oof.loc[ix]
                if sub.y.sum() < 3: continue
                diffs.append(average_precision_score(sub.y, sub[name + "|score"]) - average_precision_score(sub.y, sub[best_rule + "|score"]))
            diffs = np.array(diffs)
            paired_rows.append(dict(horizon_min=h * 5, model=name, vs=best_rule, dAUPRC=np.mean(diffs),
                                    lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5), p_le0=(diffs <= 0).mean()))
    pd.DataFrame(rows).to_csv("results/metrics_cv.csv", index=False)
    pd.DataFrame(paired_rows).to_csv("results/paired_tests.csv", index=False)

    # ---------- strict chronological hold-out ----------
    crow = []
    split = pd.Timestamp("2026-07-16")
    for h in HORIZONS:
        y, elig, eps = make_labels(d, h)
        onsets = np.array([a for a, b in eps if d.Timestamp[a] >= split])
        train_mask = (d.Timestamp < split - pd.Timedelta(days=1)).values
        test_mask = (d.Timestamp >= split).values
        out = run_split(make_models, X, y, elig, train_mask, test_mask, h, day)
        nd = day[test_mask].nunique()
        for name, (te, s, thr) in out.items():
            sm = sample_metrics(y[te], s, s >= thr)
            em = event_metrics(te, s >= thr, onsets, h, nd)
            crow.append(dict(horizon_min=h * 5, model=name, **sm, **em))
    pd.DataFrame(crow).to_csv("results/metrics_chrono.csv", index=False)
    json.dump(dict(runtime_s=time.time() - t0, n=len(d), episodes=len(eps)), open("results/run_info.json", "w"))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
