"""CropWhisperAI - reproducibility checks answering the second-round review (revision 2).

A. Inner-validation split used for each fold's alarm threshold (Section 3.8).
   run_split() in 01_train_evaluate.py first tries the chronologically last 20% of the fold's training days and
   falls back to a fixed-seed random 20% of training days when that block holds fewer than five positives.
   This script reports, for every fold and horizon, which split was used and how many positives it held.
B. BH1750 check of the on-device model (Section 4.7): full-sample AUPRCs, their simple difference, and the
   paired day-block bootstrap difference reported in the paper (read from results/revision/bh1750_lr8.csv,
   produced by `python 07_revision_analyses.py bh1750`).
C. Features of the exported model: firmware v1.0 (submitted paper) vs v1.1 (esp32/model_meta.json).

Run:  python 09_reproducibility_checks.py          (needs data.csv; part B also needs 07's output)
Output: results/revision/reproducibility_checks.json
"""
import json, os
import numpy as np, pandas as pd
from cw_common import *

SEED = 42
OUT = "results/revision"; os.makedirs(OUT, exist_ok=True)

# Features of the first export (firmware v1.0), as listed in the submitted manuscript.
V10_FEATURES = ["Tair", "Tair_mean60", "Tair_max60", "Tair_dm30_3h", "Tsoil_mean60", "Tsoil_max60", "VPD_std60", "Lux_d60"]


def inner_split_report(d):
    day = d.Timestamp.dt.normalize()
    fold, chunks = day_blocks(d, 10)
    rows = []
    for h in (3, 6, 12):
        y, elig, eps = make_labels(d, h)
        for k in range(10):
            emb = set()
            for td in chunks[k]:
                emb.update({td - np.timedelta64(1, "D"), td + np.timedelta64(1, "D")})
            tr = np.where((fold != k) & elig & ~day.isin(emb).values)[0]
            # identical to run_split() in 01_train_evaluate.py
            tr_days = np.array(sorted(day.iloc[tr].unique()))
            cut = tr_days[int(len(tr_days) * 0.8)]
            chrono_va = tr[day.iloc[tr].values >= cut]
            chrono_pos = int(y[chrono_va].sum())
            fallback = chrono_pos < 5
            if fallback:
                rng = np.random.RandomState(SEED)
                vd = rng.choice(tr_days, max(1, len(tr_days) // 5), replace=False)
                va = tr[day.iloc[tr].isin(vd).values]
            else:
                va = chrono_va
            rows.append(dict(horizon_min=h * 5, fold=k + 1, training_days=len(tr_days),
                             chronological_block_positives=chrono_pos,
                             split_used="random 20% of days (seed 42)" if fallback else "chronological last 20%",
                             inner_validation_days=int(day.iloc[va].nunique()), inner_validation_positives=int(y[va].sum())))
    return pd.DataFrame(rows)


def bh1750_report():
    p = f"{OUT}/bh1750_lr8.csv"
    if not os.path.exists(p):
        return None
    b = pd.read_csv(p); out = []
    for h, g in b.groupby("horizon_min"):
        w = g[g.model.str.startswith("With")].iloc[0]; wo = g[g.model.str.startswith("Without")].iloc[0]
        out.append(dict(horizon_min=int(h), AUPRC_with_BH1750=round(float(w.AUPRC), 4), AUPRC_without=round(float(wo.AUPRC), 4),
                        simple_difference=round(float(w.AUPRC - wo.AUPRC), 4),
                        paired_bootstrap_difference=round(float(w.dAUPRC_vs_without), 4),
                        paired_CI95=[round(float(w.d_lo), 4), round(float(w.d_hi), 4)]))
    return out


def main():
    d = load("data.csv")
    rep = inner_split_report(d)
    rep.to_csv(f"{OUT}/inner_validation_splits.csv", index=False)
    print(rep.to_string(index=False))
    n_fb = int((rep.split_used != "chronological last 20%").sum())
    print(f"\nRandom-day fallback used in {n_fb} of {len(rep)} fold x horizon combinations.")

    bh = bh1750_report()
    if bh:
        print("\nBH1750 check of the on-device model:")
        for r in bh: print(r)

    meta = json.load(open("esp32/model_meta.json"))
    v11 = meta["features"]
    feat = dict(v1_0=V10_FEATURES, v1_1=v11,
                removed=[f for f in V10_FEATURES if f not in v11], added=[f for f in v11 if f not in V10_FEATURES])
    print(f"\nExported features changed v1.0 -> v1.1: removed {feat['removed']}, added {feat['added']}")

    json.dump(dict(fallback_used=n_fb, combinations=len(rep), bh1750=bh, exported_features=feat),
              open(f"{OUT}/reproducibility_checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
