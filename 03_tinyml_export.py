"""CropWhisperAI - on-device (ESP32) early-warning model: size/accuracy trade-off and C export.

Steps
 1. Blocked-CV comparison of compact candidates on the 30-min horizon (feature selection inside each fold).
 2. Train the chosen compact model on all data, export a self-contained C header (features + model).
 3. Verify: compile a host C harness that streams the full CSV through the C code and compare with Python.
Outputs: results/tinyml_candidates.csv, results/tinyml_verify.json, esp32/cropwhisper_model.h, esp32/*.ino
Run:  python 03_tinyml_export.py
"""
import os, json, subprocess, warnings, re
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from cw_common import *

warnings.filterwarnings("ignore")
SEED, H = 42, 6
os.makedirs("results", exist_ok=True); os.makedirs("esp32", exist_ok=True)

# features that the streaming C code can compute (sensors: DHT22 + DS18B20 + BH1750)
STREAMABLE = None  # filled in main


def big_lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.7, scale_pos_weight=20, random_state=SEED, verbose=-1)


def tiny_lgbm(nt, nl):
    # no class re-weighting and a moderate learning rate: heavy weighting saturates tiny ensembles (tied scores)
    return lgb.LGBMClassifier(n_estimators=nt, learning_rate=0.2 if nt <= 25 else 0.1, num_leaves=nl,
                              min_child_samples=30, random_state=SEED, verbose=-1)


def select_top(X, y, k):
    m = big_lgbm().fit(X, y)
    g = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return list(g.index[:k])


def n_params_lgbm(m):
    dm = m.booster_.dump_model()
    return sum(t["num_leaves"] * 2 - 1 for t in dm["tree_info"])


def quantize_mlp(mlp):
    """weight-only symmetric int8 per-layer quantization (returns de-quantized copy of coefs)."""
    q = []
    for W in mlp.coefs_:
        s = np.abs(W).max() / 127.0
        q.append((np.round(W / s).clip(-127, 127) * s))
    return q


def mlp_forward(mlp, coefs, X):
    a = X
    for i, (W, b) in enumerate(zip(coefs, mlp.intercepts_)):
        a = a @ W + b
        if i < len(coefs) - 1: a = np.maximum(a, 0)
    return a[:, 0]   # logit (monotone in probability; avoids float saturation ties)


def candidates():
    c = []
    for k in (8, 12):
        for nt, nl in ((10, 4), (25, 4), (25, 8), (50, 8)):
            c.append((f"LightGBM {nt}x{nl} leaves, {k} feat", "lgb", k, (nt, nl)))
        c.append((f"Logistic regression, {k} feat", "lr", k, None))
        c.append((f"MLP 16-8 float32, {k} feat", "mlp", k, "f32"))
        c.append((f"MLP 16-8 int8 weights, {k} feat", "mlp", k, "int8"))
    return c


def main():
    d = load("data.csv")
    X = build_features(d)
    cols_ok = cols_for(["DHT22", "DS18B20", "BH1750"], X.columns)
    X = X[cols_ok]
    y, elig, eps = make_labels(d, H)
    day = d.Timestamp.dt.normalize()
    fold, chunks = day_blocks(d, 10)

    # ---- 1. blocked CV of compact candidates ----
    store = {c[0]: [] for c in candidates()}
    for k in range(10):
        emb = set()
        for td in chunks[k]:
            emb.update({td - np.timedelta64(1, "D"), td + np.timedelta64(1, "D")})
        te = np.where((fold == k) & elig)[0]
        tr = np.where((fold != k) & elig & ~day.isin(emb).values)[0]
        sel = {kk: select_top(X.iloc[tr], y[tr], kk) for kk in (8, 12)}
        for name, kind, kk, cfg in candidates():
            cols = sel[kk]
            Xtr, Xte = X.iloc[tr][cols].fillna(0), X.iloc[te][cols].fillna(0)
            if kind == "lgb":
                m = tiny_lgbm(*cfg).fit(Xtr, y[tr]); s = m.predict(Xte, raw_score=True)   # margins avoid ties
            else:
                sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
                if kind == "lr":
                    m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(A, y[tr]); s = m.decision_function(B)
                else:
                    pos = np.where(y[tr] == 1)[0]; rep = np.concatenate([np.arange(len(tr))] + [pos] * 19)  # oversample positives x20
                    m = MLPClassifier((16, 8), alpha=1e-3, max_iter=200, random_state=SEED).fit(A[rep], y[tr][rep])
                    s = mlp_forward(m, quantize_mlp(m) if cfg == "int8" else m.coefs_, B)
            store[name].append(pd.DataFrame({"idx": te, "s": s}))
        print("fold", k, "done", flush=True)

    rows = []
    for name, kind, kk, cfg in candidates():
        p = pd.concat(store[name]); yt = y[p.idx]
        if kind == "lgb":
            nparam = cfg[0] * (2 * cfg[1] - 1)   # upper bound: nodes
            size_b = cfg[0] * (cfg[1] - 1) * (1 + 4) + cfg[0] * cfg[1] * 4   # feature idx(1B)+thr(4B) per split + leaf(4B)
        elif kind == "lr":
            nparam = kk + 1 + 2 * kk; size_b = nparam * 4
        else:
            nparam = kk * 16 + 16 + 16 * 8 + 8 + 8 + 1 + 2 * kk
            size_b = (kk * 16 + 16 * 8 + 8) * (1 if cfg == "int8" else 4) + (16 + 8 + 1 + 2 * kk) * 4
        rows.append(dict(model=name, kind=kind, n_features=kk, parameters=nparam, model_bytes=size_b,
                         AUPRC=average_precision_score(yt, p.s), AUROC=roc_auc_score(yt, p.s)))
        print(rows[-1], flush=True)
    cand = pd.DataFrame(rows); cand.to_csv("results/tinyml_candidates.csv", index=False)

    # ---- 2. choose the smallest LightGBM within 0.02 AUPRC of the best compact model; train on all data ----
    best = cand.AUPRC.max()
    chosen = cand[cand.AUPRC >= best - 0.02].sort_values("model_bytes").iloc[0]
    kk = int(chosen.n_features)
    idx = np.where(elig)[0]
    feats = select_top(X.iloc[idx], y[idx], kk)
    if chosen.kind == "lgb":
        nt, nl = map(int, re.findall(r"(\d+)x(\d+)", chosen.model)[0])
        final = ("lgb", tiny_lgbm(nt, nl).fit(X.iloc[idx][feats].fillna(0), y[idx]), None)
    elif chosen.kind == "lr":
        sc = StandardScaler().fit(X.iloc[idx][feats].fillna(0))
        lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(sc.transform(X.iloc[idx][feats].fillna(0)), y[idx])
        final = ("lr", lr, sc)
    else:
        raise SystemExit("MLP export not implemented; pick an LGB or LR candidate")
    # threshold: best F1 on out-of-fold scores of the chosen configuration
    p = pd.concat(store[chosen.model]); yt = y[p.idx]   # scores are logits for every candidate kind
    qs = np.unique(np.quantile(p.s, np.linspace(0.5, 0.999, 400)))
    f1s = [2 * ((p.s >= t) & (yt == 1)).sum() / max(2 * ((p.s >= t) & (yt == 1)).sum() + ((p.s >= t) & (yt == 0)).sum() + ((p.s < t) & (yt == 1)).sum(), 1) for t in qs]
    thr = float(qs[int(np.argmax(f1s))])
    write_header(final, feats, thr)
    write_sketch()
    # ---- 3. verification harness ----
    verify(d, X, final, feats, thr, chosen)


# ---------------------------------------------------------------- C generation
FEAT_C = {}  # feature name -> C expression using helper functions


def c_expr(f):
    S = {"Tair": 0, "RH": 1, "Tsoil": 2, "Lux": 3}
    if f in ("hour_sin", "hour_cos"):
        return f"{'sinf' if f.endswith('sin') else 'cosf'}(2.0f*PI_F*hour/24.0f)"
    if f == "Tair_margin": return "(AIR_THR - cur(0))"
    if f == "Tsoil_margin": return "(SOIL_THR - cur(2))"
    if f == "Tair_dm30_3h": return "dm30_3h()"
    if f == "Lux_int_2h": return "lux_int_2h()"
    if f == "Tsoil_max_24h": return "rmax(2, 288, 12)"
    if f == "Tair_max_24h": return "rmax(0, 288, 12)"
    parts = f.split("_")
    base = parts[0]
    ch = {"Tair": 0, "RH": 1, "Tsoil": 2, "Lux": 3, "VPD": 4, "dTas": 5}[base]
    if len(parts) == 1: return f"val({ch},0)"
    op = parts[1]
    if op.startswith("d"): lag = int(op[1:]) // 5; return f"(val({ch},0) - val({ch},{lag}))"
    if op == "mean60": return f"rmean({ch},12,6)"
    if op == "std60": return f"rstd({ch},12,6)"
    if op == "max60": return f"rmax({ch},12,6)"
    raise ValueError(f)


def write_header(model, feats, thr):
    lines = []
    lines.append("// Auto-generated by 03_tinyml_export.py - CropWhisperAI heat-stress early warning (30-min horizon)")
    lines.append("// Streaming features + LightGBM model. No dynamic memory. Call cw_push() every 5 min, then cw_predict().")
    lines.append("#pragma once\n#include <math.h>\n#include <string.h>")
    lines.append("#define PI_F 3.14159265f\n#define AIR_THR 35.0f\n#define SOIL_THR 32.0f")
    lines.append(f"#define CW_ALARM_MARGIN {float(thr)!r}f  // logit threshold")
    lines.append(f"#define CW_N_FEATURES {len(feats)}")
    lines.append("#define CW_BUF 288  // 24 h of 5-min samples")
    lines.append("// channels: 0 Tair, 1 RH, 2 Tsoil, 3 Lux, 4 VPD(derived), 5 Tair-Tsoil(derived)")
    lines.append("static float cw_buf[6][CW_BUF]; static int cw_n = 0, cw_head = -1;")
    lines.append("""
static inline float cw_vpd(float t, float rh){ return 0.6108f*expf(17.27f*t/(t+237.3f))*(1.0f-rh/100.0f); }
static inline void cw_push(float tair, float rh, float tsoil, float lux){
  cw_head = (cw_head + 1) % CW_BUF; if (cw_n < CW_BUF) cw_n++;
  cw_buf[0][cw_head]=tair; cw_buf[1][cw_head]=rh; cw_buf[2][cw_head]=tsoil; cw_buf[3][cw_head]=lux;
  cw_buf[4][cw_head]=cw_vpd(tair,rh); cw_buf[5][cw_head]=tair-tsoil;
}
// value 'lag' samples ago (NAN if not yet available)
static inline float val(int ch, int lag){ if (lag >= cw_n) return NAN; return cw_buf[ch][(cw_head - lag + CW_BUF) % CW_BUF]; }
static inline float cur(int ch){ return val(ch,0); }
static inline float rmean(int ch, int w, int minp){ int m = w < cw_n ? w : cw_n; if (m < minp) return NAN; float s=0; for(int i=0;i<m;i++) s+=val(ch,i); return s/m; }
static inline float rstd(int ch, int w, int minp){ int m = w < cw_n ? w : cw_n; if (m < minp || m < 2) return NAN; float mu=rmean(ch,w,minp), s=0; for(int i=0;i<m;i++){ float e=val(ch,i)-mu; s+=e*e; } return sqrtf(s/(m-1)); }
static inline float rmax(int ch, int w, int minp){ int m = w < cw_n ? w : cw_n; if (m < minp) return NAN; float x=val(ch,0); for(int i=1;i<m;i++){ float v=val(ch,i); if(v>x) x=v; } return x; }
static inline float dm30_3h(void){ int m = 36 < cw_n ? 36 : cw_n; float s=0; for(int i=0;i<m;i++){ float e=val(0,i)-30.0f; if(e>0) s+=e; } return s*5.0f; }
static inline float lux_int_2h(void){ int m = 24 < cw_n ? 24 : cw_n; float s=0; for(int i=0;i<m;i++) s+=val(3,i); return s*5.0f*60.0f/1e6f; }
""")
    lines.append("static inline void cw_features(float hour, float *f){")
    for i, fn in enumerate(feats):
        lines.append(f"  f[{i}] = {c_expr(fn)};  // {fn}")
    lines.append("}")
    kind, mdl, sc = model
    L = lambda v: repr(float(v)) + "f"
    if kind == "lgb":
        dm = mdl.booster_.dump_model()
        def node(n, ind):
            if "leaf_value" in n:
                return f"{ind}return {L(n['leaf_value'])};\n"
            fi, t_ = n["split_feature"], L(n["threshold"])
            if n.get("missing_type", "None") == "NaN":
                cond = f"(isnan(f[{fi}]) ? {1 if n['default_left'] else 0} : (f[{fi}] <= {t_}))"
            else:
                cond = f"((isnan(f[{fi}]) ? 0.0f : f[{fi}]) <= {t_})"
            return (f"{ind}if ({cond}) {{\n" + node(n["left_child"], ind + "  ") + f"{ind}}} else {{\n"
                    + node(n["right_child"], ind + "  ") + f"{ind}}}\n")
        for t in dm["tree_info"]:
            lines.append(f"static float cw_tree{t['tree_index']}(const float *f){{\n" + node(t["tree_structure"], "  ") + "}")
        body = "  float r = 0.0f;\n" + "".join(f"  r += cw_tree{t['tree_index']}(f);\n" for t in dm["tree_info"])
    else:
        w = mdl.coef_[0] / sc.scale_; b = mdl.intercept_[0] - np.sum(mdl.coef_[0] * sc.mean_ / sc.scale_)
        lines.append("// logistic regression with standardisation folded into the weights")
        lines.append("static const float CW_W[CW_N_FEATURES] = {" + ", ".join(L(v) for v in w) + "};")
        lines.append(f"static const float CW_B = {L(b)};")
        body = "  float r = CW_B;\n  for (int i = 0; i < CW_N_FEATURES; i++) r += CW_W[i] * (isnan(f[i]) ? 0.0f : f[i]);\n"
    lines.append("// returns the logit (risk margin); alarm when >= CW_ALARM_MARGIN")
    lines.append("static inline float cw_margin(float hour){\n  float f[CW_N_FEATURES]; cw_features(hour, f);\n" + body + "  return r;\n}")
    lines.append("static inline float cw_predict(float hour){ return 1.0f/(1.0f+expf(-cw_margin(hour))); }")
    open("esp32/cropwhisper_model.h", "w").write("\n".join(lines) + "\n")
    json.dump(dict(model=kind, features=feats, logit_threshold=float(thr)), open("esp32/model_meta.json", "w"), indent=1)


def write_sketch():
    if os.path.exists("esp32/CropWhisper_EarlyWarning.ino"):
        return  # keep the maintained field firmware; only create a minimal sketch if none exists
    open("esp32/CropWhisper_EarlyWarning.ino", "w").write(r'''// CropWhisperAI - ESP32 heat-stress early warning (30 min ahead)
// Libraries: DHT sensor library (Adafruit), OneWire, DallasTemperature, BH1750 (claws), Adafruit SSD1306 + GFX
// Pins below are examples - change to match your wiring.
#include <Wire.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BH1750.h>
#include <Adafruit_SSD1306.h>
#include <time.h>
#include "cropwhisper_model.h"

#define DHT_PIN 4
#define ONEWIRE_PIN 5
DHT dht(DHT_PIN, DHT22);
OneWire ow(ONEWIRE_PIN); DallasTemperature ds(&ow);
BH1750 lightMeter;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
const uint32_t PERIOD_MS = 5UL * 60UL * 1000UL;
float lastT = NAN, lastRH = NAN, lastTs = NAN, lastLux = NAN;

void setup() {
  Serial.begin(115200); Wire.begin();
  dht.begin(); ds.begin(); lightMeter.begin();
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  configTime(19800, 0, "pool.ntp.org");   // IST; requires WiFi to be connected elsewhere in your code
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last < PERIOD_MS && last != 0) return;
  last = millis();
  float t = dht.readTemperature(), rh = dht.readHumidity();
  ds.requestTemperatures(); float ts = ds.getTempCByIndex(0);
  float lux = lightMeter.readLightLevel();
  // sensor fault -> carry last good value (same rule as the training pipeline)
  if (isnan(t) || isnan(rh)) { t = lastT; rh = lastRH; } else { lastT = t; lastRH = rh; }
  if (ts == DEVICE_DISCONNECTED_C) ts = lastTs; else lastTs = ts;
  if (lux < 0) lux = lastLux; else lastLux = lux;
  cw_push(t, rh, ts, lux);

  struct tm ti; float hour = 12.0f;
  if (getLocalTime(&ti, 50)) hour = ti.tm_hour + ti.tm_min / 60.0f;
  uint32_t t0 = micros();
  float p = cw_predict(hour);
  uint32_t dt_us = micros() - t0;
  bool warn = cw_margin(hour) >= CW_ALARM_MARGIN;

  Serial.printf("T=%.1f RH=%.1f Ts=%.1f Lux=%.0f  p=%.3f  %s  (%lu us)\n", t, rh, ts, lux, p, warn ? "WARN" : "ok", dt_us);
  oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0,0);
  oled.printf("CropWhisperAI\nAir %.1fC %.0f%%\nSoil %.1fC\nLight %.0f lx\nRisk30 %.0f%%\n%s", t, rh, ts, lux, p*100, warn ? "HEAT WARNING <30min" : "Normal");
  oled.display();
}
''')


def verify(d, X, model, feats, thr, chosen):
    # stream the whole dataset through the C implementation on the host
    hour = (d.Timestamp.dt.hour + d.Timestamp.dt.minute / 60).values
    arr = np.column_stack([d[AIR_T], d[RH], d[SOIL_T], d[LUX], hour])
    np.savetxt("esp32/_stream.txt", arr, fmt="%.6f")
    open("esp32/_harness.c", "w").write(r'''#include <stdio.h>
#include "cropwhisper_model.h"
int main(){ FILE *f=fopen("esp32/_stream.txt","r"); float a,b,c,e,h; while(fscanf(f,"%f %f %f %f %f",&a,&b,&c,&e,&h)==5){ cw_push(a,b,c,e); printf("%.7f\n", cw_margin(h)); } return 0; }
''')
    subprocess.run(["gcc", "-O2", "-I", "esp32", "esp32/_harness.c", "-o", "esp32/_harness", "-lm"], check=True)
    out = subprocess.run(["./esp32/_harness"], capture_output=True, text=True, check=True).stdout.split()
    pc = np.array(out, float)
    kind, mdl, sc = model
    Xf = X[feats].fillna(0)
    py = mdl.predict(Xf, raw_score=True) if kind == "lgb" else mdl.decision_function(sc.transform(Xf))
    y, elig, _ = make_labels(d, H)
    diff = np.abs(pc - py)
    agree = ((pc >= thr) == (py >= thr))[elig].mean()
    # compiled size of model+features (host x86-64 -Os object; ESP32 Xtensa size will differ)
    open("esp32/_size.c", "w").write('#include "cropwhisper_model.h"\nfloat cw_entry(float h){ return cw_margin(h); }\nvoid cw_entry_push(float a,float b,float c,float d){ cw_push(a,b,c,d); }\n')
    subprocess.run(["gcc", "-Os", "-c", "-I", "esp32", "esp32/_size.c", "-o", "esp32/_size.o"], check=True)
    sz = subprocess.run(["size", "esp32/_size.o"], capture_output=True, text=True).stdout.split("\n")[1].split()
    res = dict(chosen=chosen.model, features=feats, logit_threshold=float(thr), max_abs_logit_diff=float(diff[elig].max()),
               mean_abs_logit_diff=float(diff[elig].mean()), alarm_agreement=float(agree),
               host_text_bytes=int(sz[0]), host_data_bytes=int(sz[1]), host_bss_bytes=int(sz[2]),
               ring_buffer_ram_bytes=6 * 288 * 4, cv_AUPRC=float(chosen.AUPRC), cv_AUROC=float(chosen.AUROC))
    json.dump(res, open("results/tinyml_verify.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    for fn in ("_stream.txt", "_harness", "_harness.c", "_size.c", "_size.o"):
        os.remove("esp32/" + fn)


if __name__ == "__main__":
    main()
