"""CropWhisperAI - additional analyses requested in peer review (revision 1).

Run after 01, 02, 03 and 05 (uses results/oof_H*.csv). All analyses reuse the blocked-CV protocol of script 01
(10 contiguous day blocks, 1-day embargo, decision threshold chosen on an inner chronological split of the
training days, all preprocessing fitted on training data only).

Outputs (results/revision/):
  fold_table.csv              per-fold training/test sizes, positives and test episodes (30 min)
  confusion_H{h}.csv          TP/FP/FN/TN, event recall with Clopper-Pearson CI, FA/day with day-bootstrap CI
  oof_LR8_H{h}.csv            out-of-fold scores/alarms of the on-device model
  episodes_H30.csv            per-episode outcome (warned, lead, alarm persistence) for trend rule, LR8, LightGBM
  alarm_runs_H30.csv          every alarm run of LR8 / trend / LightGBM, labelled true or false
  chrono_LR8.csv              seasonal hold-out incl. the on-device model, with day-bootstrap CIs
  chrono_episodes.csv         per-episode outcomes in the seasonal hold-out
  lomo.csv                    leave-one-month-out (June / July / Aug-Sep) evaluation
  merge_sensitivity.csv       episode counts and model results for other merge gaps / minimum durations
  threshold_sensitivity.csv   episode counts and model results for thresholds shifted by +-0.5 and +-1 C
  embargo_sensitivity.csv     results with 0, 1, 2 and 3-day embargo
  qc_sensitivity.csv          results under three quality-control strategies for flagged records
  bh1750_lr8.csv              on-device LR with vs without BH1750 features (+ paired bootstrap)
  ablation_ci.csv             LightGBM sensor ablation with 95% day-bootstrap CIs and paired differences
  seeds.csv                   seed variability of the stochastic learners (5 seeds)
  cost_analysis.csv           expected-cost comparison of operating points
  summary.json                misc. numbers quoted in the paper
"""
import os, json, importlib, warnings, sys
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
import cw_common as C
from cw_common import *

warnings.filterwarnings("ignore")
m01 = importlib.import_module("01_train_evaluate")
m03 = importlib.import_module("03_tinyml_export")
OUT = "results/revision"; os.makedirs(OUT, exist_ok=True)
SEED = 42
ONLY = set(sys.argv[1:])          # optionally run a subset of sections, e.g. python 07_revision_analyses.py lomo


def want(s): return not ONLY or s in ONLY


class LR8:
    name = "On-device LR (8 features)"
    def __init__(self, k=8): self.k = k
    def fit(self, X, y):
        self.cols = m03.select_top(X, y, self.k)
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)).fit(X[self.cols].fillna(0), y)
        return self
    def score(self, X): return self.m.decision_function(X[self.cols].fillna(0))


def cp_ci(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def embargo_masks(fold, chunks, day, k, emb_days=1):
    emb = set()
    for td in chunks[k]:
        for e in range(1, emb_days + 1):
            emb.update({td - np.timedelta64(e, "D"), td + np.timedelta64(e, "D")})
    return (fold != k) & ~day.isin(emb).values, fold == k


def blocked_cv(models_fn, X, y, elig, fold, chunks, day, h, emb_days=1):
    pooled = {}
    for k in range(len(chunks)):
        trm, tem = embargo_masks(fold, chunks, day, k, emb_days)
        out = m01.run_split(models_fn, X, y, elig, trm, tem, h, day)
        for name, (te, s, thr) in out.items():
            pooled.setdefault(name, []).append(pd.DataFrame({"idx": te, "score": s, "alarm": s >= thr, "fold": k}))
    res = {}
    for name, parts in pooled.items():
        p = pd.concat(parts).sort_values("idx").reset_index(drop=True)
        p["y"] = y[p.idx]; p["day"] = day.iloc[p.idx].values
        res[name] = p
    return res


def summarize(p, onsets, h, n_days, ci=False):
    sm = m01.sample_metrics(p.y.values, p.score.values, p.alarm.values)
    em = m01.event_metrics(p.idx.values, p.alarm.values.astype(bool), onsets, h, n_days)
    r = dict(**sm, **em)
    if ci:
        (lo, hi), _ = m01.boot_ci(p, lambda s: average_precision_score(s.y, s.score))
        r.update(AUPRC_lo=lo, AUPRC_hi=hi)
    return r


def alarm_runs(idx, alarm, onsets, h):
    """List of alarm runs (start idx, end idx, true/false)."""
    a = np.sort(idx[alarm]); runs = []
    if len(a) == 0: return runs
    s = p = a[0]
    for t in a[1:]:
        if t != p + 1:
            runs.append((s, p)); s = t
        p = t
    runs.append((s, p))
    on = np.array(onsets)
    return [(s, e, bool(((on > s) & (on <= e + h)).any())) for s, e in runs]


def per_episode(idx, alarm, onsets, h):
    a_set = set(idx[alarm]); rows = []
    for o in onsets:
        win = [t for t in range(o - h, o) if t in a_set]
        n_scored = sum(1 for t in range(o - h, o) if t in set(idx))
        if win:
            first = min(win)
            cont = all(t in a_set for t in range(first, o) if t in set(idx))   # alarm stays on until onset
            toggles = sum(1 for t in range(first + 1, o) if (t in a_set) != (t - 1 in a_set))
            rows.append(dict(onset=o, warned=1, lead_min=(o - first) * STEP_MIN, alarm_fraction=len(win) / max(n_scored, 1),
                             continuous_to_onset=int(cont), toggles=toggles))
        else:
            rows.append(dict(onset=o, warned=0, lead_min=0, alarm_fraction=0.0, continuous_to_onset=0, toggles=0))
    return pd.DataFrame(rows)


def fa_boot(p, onsets, h, n=1000):
    """Day-bootstrap CI of false alarms per day (alarm runs attributed to the day they start)."""
    runs = alarm_runs(p.idx.values, p.alarm.values.astype(bool), onsets, h)
    dmap = dict(zip(p.idx.values, p.day.values))
    fa_day = pd.Series([dmap[s] for s, e, ok in runs if not ok]).value_counts()
    days = np.array(sorted(p.day.unique()))
    v = np.array([fa_day.get(dd, 0) for dd in days])
    rng = np.random.RandomState(SEED)
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def lgbm_model(seed=SEED):
    return m01.SK("LightGBM", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                                 scale_pos_weight=20, random_state=seed, verbose=-1))


def core_models(h):
    return [m01.RuleTrend(h),
            m01.SK("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))),
            lgbm_model(), LR8()]


def labels_with(d, h, air_thr=AIR_THR, soil_thr=SOIL_THR, gap=MERGE_GAP, min_len=MIN_LEN):
    s = ((d[AIR_T] > air_thr) | (d[SOIL_T] > soil_thr)).astype(int).values
    eps = episodes(s, gap, min_len)
    n = len(d); in_ep = np.zeros(n, bool); y = np.zeros(n, int)
    for a, b in eps:
        in_ep[a:b + 1] = True; y[max(0, a - h):a] = 1
    elig = (d["valid"].values == 1) & (s == 0) & (~in_ep)
    return y, elig, eps


def main():
    d = load("data.csv"); X = build_features(d)
    Xs = X[cols_for(["DHT22", "DS18B20", "BH1750"], X.columns)]
    day = d.Timestamp.dt.normalize(); fold, chunks = day_blocks(d, 10); ndays = day.nunique()
    summ = {"qc_mode": C.QC_MODE}

    # ---------- fold table ----------
    if want("folds"):
        y, elig, eps = make_labels(d, 6); on = np.array([a for a, b in eps]); rows = []
        for k in range(10):
            trm, tem = embargo_masks(fold, chunks, day, k)
            tr, te = np.where(trm & elig)[0], np.where(tem & elig)[0]
            rows.append(dict(fold=k + 1, test_first_day=str(pd.Timestamp(chunks[k][0]).date()), test_last_day=str(pd.Timestamp(chunks[k][-1]).date()),
                             test_days=len(chunks[k]), train_steps=len(tr), train_pos=int(y[tr].sum()),
                             train_episodes=int(np.isin(on, np.where(trm)[0]).sum()), test_steps=len(te), test_pos=int(y[te].sum()),
                             test_episodes=int(np.isin(on, np.where(tem)[0]).sum())))
        pd.DataFrame(rows).to_csv(f"{OUT}/fold_table.csv", index=False); print(pd.DataFrame(rows))
        dd = d.Timestamp.dt.normalize()
        ep_days = sorted({dd[a] for a, b in eps})
        summ.update(n_episode_days=len(ep_days), episodes_per_month=pd.Series([d.Timestamp[a].month for a, b in eps]).value_counts().sort_index().to_dict(),
                    n_scored=int(elig.sum()), qc_flag_counts=d.Sensor_Status.value_counts().to_dict())

    # ---------- on-device LR8 out-of-fold (all horizons) + confusion / event uncertainty ----------
    if want("confusion"):
        for h in (3, 6, 12):
            y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps])
            p = blocked_cv(lambda hh: [LR8()], Xs, y, elig, fold, chunks, day, h)[LR8.name]
            p.to_csv(f"{OUT}/oof_LR8_H{h*5}.csv", index=False)
            oof = pd.read_csv(f"results/oof_H{h*5}.csv")
            names = [c.split("|")[0] for c in oof.columns if c.endswith("|score")]
            rows = []
            for name in names + [LR8.name]:
                if name == LR8.name:
                    q = p
                else:
                    q = pd.DataFrame({"idx": oof.idx, "score": oof[name + "|score"], "alarm": oof[name + "|alarm"].astype(bool), "y": oof.y, "day": pd.to_datetime(oof.day)})
                yy, al = q.y.values, q.alarm.values.astype(bool)
                tp, fp, fn, tn = int((al & (yy == 1)).sum()), int((al & (yy == 0)).sum()), int((~al & (yy == 1)).sum()), int((~al & (yy == 0)).sum())
                em = m01.event_metrics(q.idx.values, al, on, h, ndays)
                lo, hi = cp_ci(em["detected"], em["events"]); flo, fhi = fa_boot(q, on, h)
                runs = alarm_runs(q.idx.values, al, on, h)
                rows.append(dict(horizon_min=h * 5, model=name, TP=tp, FP=fp, FN=fn, TN=tn, AUPRC=average_precision_score(yy, q.score),
                                 detected=em["detected"], events=em["events"], event_recall=em["event_recall"], event_recall_lo=lo, event_recall_hi=hi,
                                 false_alarm_runs=sum(1 for r in runs if not r[2]), FA_per_day=em["false_alarms_per_day"], FA_lo=flo, FA_hi=fhi,
                                 missed=em["events"] - em["detected"], median_lead=em["median_lead_min"],
                                 FPR_steps=fp / max(fp + tn, 1)))
            pd.DataFrame(rows).to_csv(f"{OUT}/confusion_H{h*5}.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())

    # ---------- per-episode outcomes and alarm runs, 30 min ----------
    if want("episodes"):
        h = 6; y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps])
        oof = pd.read_csv(f"results/oof_H30.csv"); p8 = pd.read_csv(f"{OUT}/oof_LR8_H30.csv")
        srcs = {"Trend rule": (oof.idx.values, oof["Rule: 30-min trend extrapolation|alarm"].values.astype(bool)),
                "LightGBM": (oof.idx.values, oof["LightGBM|alarm"].values.astype(bool)),
                "Logistic regression": (oof.idx.values, oof["Logistic regression|alarm"].values.astype(bool)),
                "On-device LR": (p8.idx.values, p8.alarm.values.astype(bool))}
        allep, allruns = [], []
        for name, (ix, al) in srcs.items():
            e = per_episode(ix, al, on, h); e["model"] = name
            e["onset_time"] = d.Timestamp.iloc[e.onset].values
            e["episode_hours"] = [(b - a + 1) * STEP_MIN / 60 for a, b in eps]
            e["Tair_onset"] = d[AIR_T].iloc[e.onset].values; e["Tsoil_onset"] = d[SOIL_T].iloc[e.onset].values
            e["trigger"] = ["air" if d[AIR_T].iloc[o] > AIR_THR else "soil" for o in e.onset]
            allep.append(e)
            for s, t, ok in alarm_runs(ix, al, on, h):
                allruns.append(dict(model=name, start=d.Timestamp.iloc[s], end=d.Timestamp.iloc[t], length_min=(t - s + 1) * STEP_MIN, true_alarm=int(ok),
                                    hour=d.Timestamp.iloc[s].hour, Tair=d[AIR_T].iloc[s], Tsoil=d[SOIL_T].iloc[s]))
        E = pd.concat(allep); E.to_csv(f"{OUT}/episodes_H30.csv", index=False)
        R = pd.DataFrame(allruns); R.to_csv(f"{OUT}/alarm_runs_H30.csv", index=False)
        for name, g in E.groupby("model"):
            w = g[g.warned == 1]
            summ[f"lead_{name}"] = dict(n_warned=int(len(w)), lead_quartiles=list(np.percentile(w.lead_min, [0, 25, 50, 75, 100])),
                                        frac_lead_ge10=float((w.lead_min >= 10).mean()), frac_lead_ge15=float((w.lead_min >= 15).mean()),
                                        frac_lead_ge20=float((w.lead_min >= 20).mean()), frac_lead_30=float((w.lead_min >= 30).mean()),
                                        frac_continuous=float(w.continuous_to_onset.mean()), median_toggles=float(w.toggles.median()),
                                        max_toggles=int(w.toggles.max()), missed=g[g.warned == 0].onset_time.astype(str).tolist(),
                                        missed_trigger=g[g.warned == 0].trigger.tolist())
        for name, g in R.groupby("model"):
            summ[f"runs_{name}"] = dict(n_true=int(g.true_alarm.sum()), n_false=int((1 - g.true_alarm).sum()),
                                        false_dates=g[g.true_alarm == 0].start.astype(str).tolist(),
                                        false_len_median=float(g[g.true_alarm == 0].length_min.median()) if (g.true_alarm == 0).any() else None,
                                        true_len_median=float(g[g.true_alarm == 1].length_min.median()))
        print(json.dumps({k: v for k, v in summ.items() if k.startswith(("lead_", "runs_"))}, indent=1, default=str))

    # ---------- seasonal hold-out incl. on-device model ----------
    if want("chrono"):
        split = pd.Timestamp("2026-07-16"); crow, cep = [], []
        for h in (3, 6, 12):
            y, elig, eps = make_labels(d, h)
            on = np.array([a for a, b in eps if d.Timestamp[a] >= split])
            trm = (d.Timestamp < split - pd.Timedelta(days=1)).values; tem = (d.Timestamp >= split).values
            fns = m01.make_models
            out = m01.run_split(fns, X, y, elig, trm, tem, h, day)
            # LR8 must use streamable columns only -> refit separately
            out[LR8.name] = m01.run_split(lambda hh: [LR8()], Xs, y, elig, trm, tem, h, day)[LR8.name]
            nd = day[tem].nunique()
            for name, (te, s, thr) in out.items():
                p = pd.DataFrame({"idx": te, "score": s, "alarm": s >= thr, "y": y[te], "day": day.iloc[te].values})
                r = summarize(p, on, h, nd, ci=True); flo, fhi = fa_boot(p, on, h)
                lo, hi = cp_ci(r["detected"], r["events"])
                crow.append(dict(horizon_min=h * 5, model=name, **r, FA_lo=flo, FA_hi=fhi, event_recall_lo=lo, event_recall_hi=hi,
                                 threshold=thr, train_episodes=int(sum(1 for a, b in eps if d.Timestamp[a] < split - pd.Timedelta(days=1)))))
                if h == 6 and name in ("Logistic regression", "LightGBM", "Rule: 30-min trend extrapolation", LR8.name):
                    e = per_episode(p.idx.values, p.alarm.values, on, h); e["model"] = name; e["onset_time"] = d.Timestamp.iloc[e.onset].values
                    cep.append(e)
        pd.DataFrame(crow).to_csv(f"{OUT}/chrono_LR8.csv", index=False); pd.concat(cep).to_csv(f"{OUT}/chrono_episodes.csv", index=False)
        print(pd.DataFrame(crow)[["horizon_min", "model", "AUPRC", "AUPRC_lo", "AUPRC_hi", "detected", "events", "median_lead_min", "false_alarms_per_day", "FA_lo", "FA_hi"]].round(3).to_string())

    # ---------- leave-one-month-out ----------
    if want("lomo"):
        h = 6; y, elig, eps = make_labels(d, h); mon = d.Timestamp.dt.month.values
        blocks = {"June": mon == 6, "July": mon == 7, "Aug-Sep": mon >= 8}; rows = []
        for bname, tem in blocks.items():
            tdays = day[tem]; first, last = tdays.min(), tdays.max()
            trm = (~tem) & ~day.isin([first - pd.Timedelta(days=1), last + pd.Timedelta(days=1)]).values
            on = np.array([a for a, b in eps if tem[a]])
            out = m01.run_split(lambda hh: [m01.RuleTrend(hh), m01.SK("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))), lgbm_model()],
                                X, y, elig, trm, tem, h, day)
            out[LR8.name] = m01.run_split(lambda hh: [LR8()], Xs, y, elig, trm, tem, h, day)[LR8.name]
            for name, (te, s, thr) in out.items():
                p = pd.DataFrame({"idx": te, "score": s, "alarm": s >= thr, "y": y[te], "day": day.iloc[te].values})
                r = summarize(p, on, h, day[tem].nunique())
                rows.append(dict(test_block=bname, model=name, test_days=day[tem].nunique(), test_episodes=len(on), train_episodes=len(eps) - len(on), **r))
        L = pd.DataFrame(rows); L.to_csv(f"{OUT}/lomo.csv", index=False)
        print(L[["test_block", "model", "test_episodes", "AUPRC", "detected", "median_lead_min", "false_alarms_per_day"]].round(3).to_string())

    # ---------- merge-gap / minimum-duration sensitivity ----------
    if want("merge"):
        rows = []
        for gap in (1, 3, 6, 9, 12):   # max. time between thermal readings that are merged (steps of 5 min; 1 = no merging)
            for ml in (2, 3, 4):
                y, elig, eps = labels_with(d, 6, gap=gap, min_len=ml)
                r = dict(merge_gap_min=gap * 5, min_duration_min=ml * 5, episodes=len(eps), scored=int(elig.sum()),
                         median_episode_h=float(np.median([(b - a + 1) * 5 / 60 for a, b in eps])))
                if ml == 3 and gap in (1, 3, 6, 12):
                    on = np.array([a for a, b in eps])
                    res = blocked_cv(lambda hh: [m01.RuleTrend(hh), lgbm_model()], X, y, elig, fold, chunks, day, 6)
                    res.update(blocked_cv(lambda hh: [LR8()], Xs, y, elig, fold, chunks, day, 6))
                    for name, p in res.items():
                        s = summarize(p, on, 6, ndays)
                        r.update({f"{name}|AUPRC": s["AUPRC"], f"{name}|warned": f"{s['detected']}/{s['events']}", f"{name}|FA_day": s["false_alarms_per_day"],
                                  f"{name}|lead": s["median_lead_min"]})
                rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/merge_sensitivity.csv", index=False)

    # ---------- threshold sensitivity (sensor accuracy +-0.5 C, +-1 C) ----------
    if want("thresh"):
        rows = []
        for da in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for ds in (-1.0, -0.5, 0.0, 0.5, 1.0):
                y, elig, eps = labels_with(d, 6, AIR_THR + da, SOIL_THR + ds)
                r = dict(air_thr=AIR_THR + da, soil_thr=SOIL_THR + ds, episodes=len(eps),
                         thermal_steps=int(((d[AIR_T] > AIR_THR + da) | (d[SOIL_T] > SOIL_THR + ds)).sum()))
                if da == ds and da in (-1.0, -0.5, 0.5, 1.0):
                    on = np.array([a for a, b in eps])
                    C.AIR_THR, C.SOIL_THR = AIR_THR + da, SOIL_THR + ds     # trend rule uses the shifted thresholds
                    m01.AIR_THR, m01.SOIL_THR = C.AIR_THR, C.SOIL_THR
                    res = blocked_cv(lambda hh: [m01.RuleTrend(hh), lgbm_model()], X, y, elig, fold, chunks, day, 6)
                    res.update(blocked_cv(lambda hh: [LR8()], Xs, y, elig, fold, chunks, day, 6))
                    C.AIR_THR, C.SOIL_THR = AIR_THR, SOIL_THR; m01.AIR_THR, m01.SOIL_THR = AIR_THR, SOIL_THR
                    for name, p in res.items():
                        s = summarize(p, on, 6, ndays)
                        r.update({f"{name}|AUPRC": s["AUPRC"], f"{name}|warned": f"{s['detected']}/{s['events']}", f"{name}|FA_day": s["false_alarms_per_day"],
                                  f"{name}|lead": s["median_lead_min"]})
                rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/threshold_sensitivity.csv", index=False)

    # ---------- embargo sensitivity ----------
    if want("embargo"):
        h = 6; y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps]); rows = []
        for e in (0, 1, 2, 3):
            res = blocked_cv(lambda hh: [m01.RuleTrend(hh), m01.SK("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))), lgbm_model()],
                             X, y, elig, fold, chunks, day, h, emb_days=e)
            res.update(blocked_cv(lambda hh: [LR8()], Xs, y, elig, fold, chunks, day, h, emb_days=e))
            for name, p in res.items():
                rows.append(dict(embargo_days=e, model=name, **summarize(p, on, h, ndays)))
            print(e, "done", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/embargo_sensitivity.csv", index=False)
        # temporal dependence: autocorrelation of air temperature anomalies (diurnal cycle removed)
        hr = d.Timestamp.dt.hour * 12 + d.Timestamp.dt.minute // 5
        an = d[AIR_T] - d.groupby(hr)[AIR_T].transform("mean")
        summ["acf_Tair_anomaly"] = {f"{l * 5 // 60}h": float(an.autocorr(l)) for l in (12, 72, 144, 288, 576)}

    # ---------- QC strategy sensitivity ----------
    if want("qc"):
        rows = []
        for mode in ("row", "sensor", "sensor_nan"):
            dq = load("data.csv", mode=mode); Xq = build_features(dq); Xqs = Xq[cols_for(["DHT22", "DS18B20", "BH1750"], Xq.columns)]
            h = 6; y, elig, eps = make_labels(dq, h); on = np.array([a for a, b in eps])
            res = blocked_cv(lambda hh: [m01.RuleTrend(hh), m01.SK("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))), lgbm_model()],
                             Xq, y, elig, fold, chunks, day, h)
            res.update(blocked_cv(lambda hh: [LR8()], Xqs, y, elig, fold, chunks, day, h))
            for name, p in res.items():
                rows.append(dict(qc_mode=mode, model=name, scored=int(elig.sum()), positives=int(y[elig].sum()), episodes=len(eps), **summarize(p, on, h, ndays)))
            print(mode, "done", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/qc_sensitivity.csv", index=False)

    # ---------- BH1750 contribution to the on-device model ----------
    if want("bh1750"):
        rows = []; Xt = X[cols_for(["DHT22", "DS18B20"], X.columns)]
        for h in (3, 6, 12):
            y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps])
            pa = blocked_cv(lambda hh: [LR8()], Xs, y, elig, fold, chunks, day, h)[LR8.name]
            pb = blocked_cv(lambda hh: [LR8()], Xt, y, elig, fold, chunks, day, h)[LR8.name]
            # how often a BH1750 feature was selected in the folds
            for name, p in (("With BH1750 (deployed)", pa), ("Without BH1750 (DHT22 + DS18B20)", pb)):
                rows.append(dict(horizon_min=h * 5, model=name, **summarize(p, on, h, ndays, ci=True)))
            m = pa.merge(pb[["idx", "score"]], on="idx", suffixes=("", "_b")); rng = np.random.RandomState(SEED)
            g = {k: v.index.values for k, v in m.groupby("day")}; ud = m.day.unique(); diffs = []
            for _ in range(1000):
                ix = np.concatenate([g[k] for k in rng.choice(ud, len(ud))]); s = m.loc[ix]
                if s.y.sum() < 3: continue
                diffs.append(average_precision_score(s.y, s.score) - average_precision_score(s.y, s.score_b))
            rows[-2].update(dAUPRC_vs_without=np.mean(diffs), d_lo=np.percentile(diffs, 2.5), d_hi=np.percentile(diffs, 97.5))
            print(rows[-2:], flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/bh1750_lr8.csv", index=False)
        # fold-wise selection frequency of BH1750 features (30 min)
        y, elig, eps = make_labels(d, 6); cnt = {}
        for k in range(10):
            trm, _ = embargo_masks(fold, chunks, day, k); tr = np.where(trm & elig)[0]
            for c in m03.select_top(Xs.iloc[tr], y[tr], 8): cnt[c] = cnt.get(c, 0) + 1
        summ["lr8_feature_selection_frequency"] = cnt

    # ---------- ablation with CIs ----------
    if want("ablation"):
        m02 = importlib.import_module("02_ablation_explain"); rows = []; h = 6
        y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps]); store = {}
        subsets = dict(m02.SUBSETS); subsets.pop("All minus soil moisture & water level")
        for name, sens in subsets.items():
            cols = cols_for(sens, X.columns)
            p = blocked_cv(lambda hh: [lgbm_model()], X[cols], y, elig, fold, chunks, day, h)["LightGBM"]
            store[name] = p
            rows.append(dict(subset=name, n_features=len(cols), **summarize(p, on, h, ndays, ci=True)))
            print(rows[-1], flush=True)
        ref = store["All five sensors"]; rng = np.random.RandomState(SEED)
        g = {k: v.index.values for k, v in ref.groupby("day")}; ud = ref.day.unique()
        picks = [np.concatenate([g[k] for k in rng.choice(ud, len(ud))]) for _ in range(1000)]
        for r in rows:
            p = store[r["subset"]]; diffs = []
            for ix in picks:
                if ref.y.values[ix].sum() < 3: continue
                diffs.append(average_precision_score(p.y.values[ix], p.score.values[ix]) - average_precision_score(ref.y.values[ix], ref.score.values[ix]))
            r.update(dAUPRC_vs_all=np.mean(diffs), d_lo=np.percentile(diffs, 2.5), d_hi=np.percentile(diffs, 97.5))
        pd.DataFrame(rows).to_csv(f"{OUT}/ablation_ci.csv", index=False)
        # correlation among sensor channels (daily-cycle driven)
        summ["channel_corr"] = d[[AIR_T, RH, SOIL_T, SOIL_M, LUX, WL]].corr().round(2).to_dict()

    # ---------- seed variability ----------
    if want("seeds"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        h = 6; y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps]); rows = []
        for sd in (1, 7, 42, 123, 2026):
            fns = lambda hh: [lgbm_model(sd),
                              m01.SK("Random forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=sd)),
                              m01.SK("MLP (64-32)", make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=sd)))]
            for name, p in blocked_cv(fns, X, y, elig, fold, chunks, day, h).items():
                rows.append(dict(seed=sd, model=name, **summarize(p, on, h, ndays)))
            print(sd, "done", flush=True)
        S = pd.DataFrame(rows); S.to_csv(f"{OUT}/seeds.csv", index=False)
        print(S.groupby("model")[["AUPRC", "event_recall", "false_alarms_per_day"]].agg(["mean", "std", "min", "max"]).round(3))

    # ---------- cost analysis ----------
    if want("cost"):
        c = pd.read_csv(f"{OUT}/confusion_H30.csv"); rows = []
        for ratio in (1, 5, 10, 20, 50):
            for _, r in c.iterrows():
                rows.append(dict(miss_to_false_alarm_cost=ratio, model=r.model, false_alarms=r.false_alarm_runs, missed=r.missed,
                                 cost_units=r.false_alarm_runs + ratio * r.missed))
        pd.DataFrame(rows).to_csv(f"{OUT}/cost_analysis.csv", index=False)

    old = json.load(open(f"{OUT}/summary.json")) if os.path.exists(f"{OUT}/summary.json") else {}
    old.update(summ); json.dump(old, open(f"{OUT}/summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
