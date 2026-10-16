"""CropWhisperAI - event-level evaluation of the deployed on-device model (same blocked CV + inner threshold as script 01).
Feature selection (top-8 by LightGBM gain) is repeated inside every training fold. Output: results/deployed_eval.csv"""
import importlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from cw_common import *
m01 = importlib.import_module("01_train_evaluate"); m03 = importlib.import_module("03_tinyml_export")

class LR8:
    name = "On-device LR (8 features)"
    def fit(self, X, y):
        self.cols = m03.select_top(X, y, 8)
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)).fit(X[self.cols].fillna(0), y)
        return self
    def score(self, X): return self.m.decision_function(X[self.cols].fillna(0))

d = load("data.csv"); X = build_features(d); X = X[cols_for(["DHT22", "DS18B20", "BH1750"], X.columns)]
day = d.Timestamp.dt.normalize(); fold, chunks = day_blocks(d, 10); rows = []
for h in (3, 6, 12):
    y, elig, eps = make_labels(d, h); on = np.array([a for a, b in eps]); parts = []
    for k in range(10):
        emb = set()
        for td in chunks[k]: emb.update({td - np.timedelta64(1, "D"), td + np.timedelta64(1, "D")})
        out = m01.run_split(lambda hh: [LR8()], X, y, elig, (fold != k) & ~day.isin(emb).values, fold == k, h, day)
        te, s, thr = out[LR8.name]; parts.append(pd.DataFrame({"idx": te, "score": s, "alarm": s >= thr}))
    p = pd.concat(parts).sort_values("idx").reset_index(drop=True); p["y"] = y[p.idx]; p["day"] = day.iloc[p.idx].values
    sm = m01.sample_metrics(p.y.values, p.score.values, p.alarm.values)
    em = m01.event_metrics(p.idx.values, p.alarm.values, on, h, day.nunique())
    (lo, hi), _ = m01.boot_ci(p, lambda s: average_precision_score(s.y, s.score))
    rows.append(dict(horizon_min=h * 5, model=LR8.name, **sm, AUPRC_lo=lo, AUPRC_hi=hi, **em)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/deployed_eval.csv", index=False)
