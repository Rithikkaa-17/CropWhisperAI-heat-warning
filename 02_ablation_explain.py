"""CropWhisperAI - sensor-subset ablation and SHAP explanation (LightGBM, 30-min horizon by default).

Outputs (results/):
  ablation.csv        pooled blocked-CV AUPRC/AUROC for each sensor subset, all horizons
  shap_importance.csv mean |SHAP| per feature and per sensor group (30-min horizon, all data)
  shap_values.npy / shap_X.csv   for plotting
Run:  python 02_ablation_explain.py
"""
import os, warnings
import numpy as np, pandas as pd
import lightgbm as lgb, shap
from sklearn.metrics import roc_auc_score, average_precision_score
from cw_common import *

warnings.filterwarnings("ignore")
SEED = 42
os.makedirs("results", exist_ok=True)

SUBSETS = {
    "DHT22 only": ["DHT22"],
    "DS18B20 only": ["DS18B20"],
    "DHT22 + DS18B20": ["DHT22", "DS18B20"],
    "DHT22 + DS18B20 + BH1750": ["DHT22", "DS18B20", "BH1750"],
    "All five sensors": ["DHT22", "DS18B20", "BH1750", "Capacitive", "WaterLevel"],
    "All minus soil moisture & water level": ["DHT22", "DS18B20", "BH1750"],
}


def lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              scale_pos_weight=20, random_state=SEED, verbose=-1)


def blocked_oof(X, y, elig, fold, chunks, day):
    parts = []
    for k in range(len(chunks)):
        emb = set()
        for td in chunks[k]:
            emb.update({td - np.timedelta64(1, "D"), td + np.timedelta64(1, "D")})
        te = np.where((fold == k) & elig)[0]
        tr = np.where((fold != k) & elig & ~day.isin(emb).values)[0]
        m = lgbm().fit(X.iloc[tr], y[tr])
        parts.append(pd.DataFrame({"idx": te, "s": m.predict_proba(X.iloc[te])[:, 1]}))
    p = pd.concat(parts)
    return y[p.idx], p.s.values


def main():
    d = load("data.csv")
    Xall = build_features(d)
    day = d.Timestamp.dt.normalize()
    fold, chunks = day_blocks(d, 10)
    rows = []
    seen = set()
    for h in (6,):   # revision: subsets with CIs are in 07_revision_analyses.py; 30-min horizon only here
        y, elig, _ = make_labels(d, h)
        for name, sens in ({} if os.environ.get("CW_FAST") else SUBSETS).items():
            cols = cols_for(sens, Xall.columns)
            key = (h, tuple(cols))
            yt, s = blocked_oof(Xall[cols], y, elig, fold, chunks, day)
            rows.append(dict(horizon_min=h * 5, subset=name, n_features=len(cols),
                             AUPRC=average_precision_score(yt, s), AUROC=roc_auc_score(yt, s)))
            print(rows[-1], flush=True)
        # time-only reference
        yt, s = blocked_oof(Xall[["hour_sin", "hour_cos"]], y, elig, fold, chunks, day)
        rows.append(dict(horizon_min=h * 5, subset="No sensors (time of day)", n_features=2,
                         AUPRC=average_precision_score(yt, s), AUROC=roc_auc_score(yt, s)))
    pd.DataFrame(rows).drop_duplicates().to_csv("results/ablation.csv", index=False)

    # SHAP on the 30-min model trained on all eligible samples
    y, elig, _ = make_labels(d, 6)
    idx = np.where(elig)[0]
    m = lgbm().fit(Xall.iloc[idx], y[idx])
    rng = np.random.RandomState(SEED)
    samp = np.concatenate([idx[y[idx] == 1], rng.choice(idx[y[idx] == 0], 4000, replace=False)])
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(Xall.iloc[samp])
    sv = sv[1] if isinstance(sv, list) else sv
    np.save("results/shap_values.npy", sv)
    Xall.iloc[samp].to_csv("results/shap_X.csv", index=False)
    imp = pd.DataFrame({"feature": Xall.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)

    def group(c):
        p = c.split("_")[0]
        if c.startswith("hour_"): return "Time of day"
        if p == "dTas": return "DHT22 x DS18B20 (gradient)"
        for g, prefs in SENSOR_GROUPS.items():
            if p in prefs: return g
        return "other"
    imp["group"] = imp.feature.map(group)
    imp.to_csv("results/shap_importance.csv", index=False)
    print(imp.head(15).to_string())
    print(imp.groupby("group").mean_abs_shap.sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
