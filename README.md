# CropWhisperAI — on-device early warning of heat-stress episodes in tomato

Data, analysis code and ESP32 firmware for the paper

> Rithikkaa S. J., Parinitha Mohanavelu, Rahul S. G., Amirthalakshmi T. M.
> **Edge Early Warning of Threshold-Defined Heat-Stress Episodes in Tomato: A Proof-of-Concept Low-Cost ESP32
> Multi-Sensor Node with On-Device Machine Learning.** *AgriEngineering* (accepted, 2026).

A low-cost ESP32 node logged air temperature/humidity (DHT22), soil temperature (DS18B20), soil moisture
(capacitive probe), illuminance (BH1750) and reservoir level beside one field-grown tomato plant every 5 min for
107 days (1 June – 15 September 2026, 30,816 records). The scripts here test whether threshold-defined heat-stress
episodes (air > 35 °C or soil > 32 °C for at least 15 min) can be forecast 15–60 min ahead, compare rule baselines
with learned models under leakage-aware day-blocked cross-validation, and export a nine-parameter logistic model to C
for the ESP32.

**Scope.** This is a proof of concept from one plant, one site and one season. The models forecast threshold
crossings, not physiological damage. See Sections 5.5–5.6 of the paper before reusing the results.

## Repository contents

| Path | Contents |
|---|---|
| `data.csv` | The sensor record (see [Data](#data)) |
| `cw_common.py` | Data loading and quality control, episode definition, labels, the 64 causal features, day blocks |
| `01_train_evaluate.py` | Day-blocked cross-validation of rules and learned models at 15/30/60 min; seasonal hold-out; paired tests |
| `02_ablation_explain.py` | Sensor ablation and SHAP attributions |
| `03_tinyml_export.py` | Compact on-device candidates, C export (`esp32/cropwhisper_model.h`) and host verification of the C code |
| `04_figures.py`, `06_diagrams.py`, `08_revision_figures.py` | Figures of the paper |
| `05_deployed_eval.py` | Event-level results of the on-device model |
| `07_revision_analyses.py` | Analyses added in peer review: fold table, confusion counts, event CIs, lead times, seasonal hold-out incl. on-device model, leave-one-month-out, sensitivity (merge gap, thresholds, QC, embargo, seeds), ablation CIs, BH1750 test, cost analysis |
| `09_reproducibility_checks.py` | Checks for the second review round: inner-validation split per fold, paired vs simple BH1750 difference, v1.0 vs v1.1 features |
| `esp32/` | Field firmware (`CropWhisper_EarlyWarning.ino`, v1.1), generated model header, model metadata, compile report |
| `results/` | Result tables behind every number in the paper |

## Quick start

```bash
pip install -r requirements.txt
python 01_train_evaluate.py      # ~15 min on 2 CPUs
python 02_ablation_explain.py
python 03_tinyml_export.py       # needs gcc for the host verification of the C code
python 05_deployed_eval.py
python 07_revision_analyses.py   # long; run one section with e.g. `python 07_revision_analyses.py bh1750`
python 04_figures.py
python 08_revision_figures.py
python 09_reproducibility_checks.py
```

All scripts use random seed 42. Results in the paper were produced with Python 3.13, scikit-learn 1.8 and
LightGBM 4.7.

## Data

`data.csv` has one row per 5-min record, site code `VENGAL_SITE_01` (Thiruvallur district, Tamil Nadu, India).

| Column | Unit | Sensor |
|---|---|---|
| `Timestamp` | local time (IST) | ESP32 / NTP |
| `Location_ID` | – | – |
| `DHT22_Air_Temperature_C` | °C | DHT22, canopy height (≈1.0 m) |
| `DHT22_Humidity_pct` | % RH | DHT22 |
| `DS18B20_Soil_Temperature_C` | °C | DS18B20, root zone (≈5–10 cm) |
| `Capacitive_Soil_Moisture_pct` | % (relative, uncalibrated) | capacitive probe |
| `BH1750_Light_lux` | lx | BH1750 |
| `Water_Level_pct` | % (relative, uncalibrated) | resistive water-level strip |
| `Sensor_Status` | flag | firmware |

`Sensor_Status` is `OK` for 30,792 records. The 24 flagged records are isolated single records:
8 `DHT22_TIMEOUT`, 6 `WIFI_RETRY`, 6 `RECALIBRATION_CHECK` and 4 `MAINTENANCE`. The timestamp sequence has no gaps.

### Quality control

Set with the environment variable `CW_QC` (default `sensor`; all published numbers use it):

| Mode | Handling of flagged records |
|---|---|
| `sensor` | Only the affected sensor is invalidated: `DHT22_TIMEOUT` → air temperature and humidity; `MAINTENANCE`, `RECALIBRATION_CHECK` → all sensors; `WIFI_RETRY` → none (upload failure only). The last valid value of that sensor is carried forward. A time step is scored only if both temperature sensors are valid. |
| `row` | Submitted version: any non-OK flag invalidates every sensor on the row. |
| `sensor_nan` | As `sensor`, but invalid values are left missing. |

## Evaluation protocol (Section 3.8)

* **Folds.** 10 contiguous day blocks (`day_blocks()` in `cw_common.py`; dates in `results/revision/fold_table.csv`),
  each the test set once, with a one-day embargo on both sides.
* **Alarm thresholds.** In `run_split()` (`01_train_evaluate.py`), each model is fitted on inner-training days, its
  F1-maximising threshold is chosen on inner-validation days from 300 score quantiles (50th–99.9th percentile), and
  the model is refitted on all training days of the fold. Inner validation is the last 20 % of training days, or a
  fixed-seed random 20 % of training days if that block holds fewer than five positives. In this record the random
  split was used in every fold at every horizon (`results/revision/inner_validation_splits.csv`).
* **Preprocessing.** Standardisation, class weights and the on-device model's feature selection are fitted on the
  training days of each fold only.
* **Uncertainty.** Day-block bootstrap (1000 resamples) for AUPRC and false alarms per day; Clopper–Pearson
  intervals for warned episodes. Paired differences (e.g. with vs without BH1750) are the mean of the paired
  bootstrap differences, which is why they can differ slightly from the difference of two full-sample AUPRCs
  (30 min: 0.894 − 0.802 = 0.091 unrounded, paired 0.088).

## On-device model and firmware

The deployed model is a logistic regression on eight streaming features. It is a single fit on all 107 days with
one feature selection on the full record; its alarm threshold (logit 4.17) maximises F1 on the pooled out-of-fold
scores. Its accuracy is estimated by cross-validating the same procedure (feature selection repeated inside each
fold), not by scoring the exported model on its own training data.

| | Firmware v1.0 (submitted) | Firmware v1.1 (this repository) |
|---|---|---|
| Features | Tair, Tair_mean60, Tair_max60, Tair_dm30_3h, Tsoil_mean60, **Tsoil_max60**, VPD_std60, **Lux_d60** | Tair, Tair_mean60, Tair_max60, Tair_dm30_3h, Tsoil_mean60, VPD_std60, **RH_d30**, **Tair_std60** |
| Quality control | row-level | per sensor |
| Guards | – | no warning in the first 65 min after a restart; fault and no warning after more than three consecutive invalid temperature readings |

What has been verified for which version:

| Item | Verified for |
|---|---|
| Features, coefficients, threshold (`esp32/model_meta.json`) | v1.1 |
| C code vs Python: all alarm decisions agree at 26,047 time steps, max. logit difference 4.0 × 10⁻⁵ (`results/tinyml_verify.json`) | v1.1 |
| Host-compiled model code, 1095 bytes | v1.1 |
| ESP32 model code 1016 B flash / 6920 B RAM; full firmware 798,085 B flash / 53,620 B static RAM (`esp32/compile_report.txt`) | v1.0 only |
| On-board inference time 257–300 µs (`inference_us` in the serial log) | v1.0 only |

v1.1 has not been re-flashed and re-measured on hardware; the paper reports the v1.0 hardware figures with this
caveat. Pins (DHT22 GPIO 4, DS18B20 GPIO 5, soil probe GPIO 34, water level GPIO 35, BH1750 I²C SDA 21/SCL 22),
Wi-Fi credentials, upload endpoint and analogue calibration counts are set at the top of the sketch.
The firmware still reads and uploads the BH1750, but the v1.1 model does not use it.

## Where each result comes from

| Paper | File |
|---|---|
| Table 7 (folds) | `results/revision/fold_table.csv` |
| Table 8 (30-min comparison) | `results/metrics_cv.csv`, `results/revision/confusion_H30.csv` |
| Section 4.2 (horizons) | `results/metrics_cv.csv`, `results/revision/confusion_H15.csv`, `confusion_H60.csv` |
| Table 9 (seasonal hold-out) | `results/revision/chrono_LR8.csv` |
| Table 10 (leave-one-month-out) | `results/revision/lomo.csv` |
| Table 11, Section 4.5 (sensitivity, seeds) | `results/revision/{merge,threshold,qc,embargo}_sensitivity.csv`, `seeds.csv` |
| Sensor ablation, SHAP | `results/revision/ablation_ci.csv`, `results/shap_importance.csv` |
| Compact candidates (Figure 12) | `results/tinyml_candidates.csv` |
| BH1750 check, Table 12 | `results/revision/bh1750_lr8.csv`, `results/tinyml_verify.json`, `esp32/compile_report.txt` |
| Cost analysis | `results/revision/cost_analysis.csv` |

Large per-record files (`results/oof_H*.csv`, `results/shap_X.csv`, `results/shap_values.npy`) and rendered figures
are not stored here; the scripts regenerate them.

## Citation

Please cite the paper (see `CITATION.cff`). Related work by the group:
Rahul S. G., Rithikkaa S. J., Amirthalakshmi T. M., Gayathri M. Sustainable agriculture through IoT-driven smart
irrigation with explainable AI. *Front. Sustain. Food Syst.* **2026**, 10, 1847041.
https://doi.org/10.3389/fsufs.2026.1847041

## Contact

Rahul S. G. (corresponding author), Department of Electronics and Communication Engineering, Amrita School of
Engineering, Amrita Vishwa Vidyapeetham, Chennai, India — sg_rahul@ch.amrita.edu
