# CropWhisperAI heat-stress early warning

Put `data.csv` (exported from the cleaned sensor spreadsheet) in this folder, then run the scripts in order:

    pip install pandas numpy scikit-learn lightgbm shap matplotlib
    python 01_train_evaluate.py      # blocked CV of rules and models, 15/30/60 min (~15 min on 2 CPUs)
    python 02_ablation_explain.py    # sensor ablation and SHAP
    python 03_tinyml_export.py       # compact models, C export and host verification (needs gcc)
    python 05_deployed_eval.py       # event-level results of the on-device model
    python 07_revision_analyses.py   # analyses added in peer review (folds, confusion counts, event CIs, lead times,
                                     # seasonal hold-out incl. on-device model, leave-one-month-out, sensitivity to
                                     # merge gap / thresholds / QC / embargo / seeds, ablation CIs, BH1750 test)
    python 04_figures.py             # paper figures (600 dpi, Liberation Sans throughout)
    python 08_revision_figures.py    # Figures 8-9 of the revised paper (examples; lead times and persistence)

Quality control of flagged records is set by the environment variable `CW_QC` (default `sensor`):
`sensor` invalidates only the affected sensor (DHT22_TIMEOUT -> air T and RH; MAINTENANCE / RECALIBRATION_CHECK -> all;
WIFI_RETRY -> none) and carries the last valid value of that sensor forward; `row` reproduces the submitted version;
`sensor_nan` leaves invalid values missing. All published numbers use `CW_QC=sensor`.

`cw_common.py` holds the shared loading, episode definition and 64 causal features.

## ESP32 firmware
`esp32/CropWhisper_EarlyWarning.ino` is the field firmware: it reads DHT22 (GPIO 4), DS18B20 (GPIO 5), capacitive soil
probe (GPIO 34), BH1750 (I2C, SDA 21/SCL 22) and water-level strip (GPIO 35) every 5 min, flags failed readings with the
dataset's status codes, carries forward the last valid value, runs the model in `esp32/cropwhisper_model.h`, times it
with micros(), updates the OLED, prints a CSV line on Serial and uploads JSON (REST) or ThingSpeak fields.
Set Wi-Fi credentials, the endpoint/API key and the analogue calibration counts at the top of the sketch.
Firmware v1.1 (revision) adds two guards: no warning until 65 min of data are available after a restart, and no warning
(fault shown on the OLED) after more than three consecutive invalid air- or soil-temperature readings.
`esp32/compile_report.txt` gives the verified build of v1.0: 798,085 B flash, 53,620 B static RAM (arduino-esp32 2.0.17).
These v1.0 figures were carried forward in the manuscript's Table 12 with a disclosed caveat: firmware v1.1 (which drops
the illuminance feature after the quality-control correction) was not physically re-flashed and re-measured on hardware.
Inference time: take the median of the `inference_us` column of the serial log and put it in Table 5.

## Results
All numbers in the manuscript come from `results/*.csv` and `results/*.json`, and the additional peer-review analyses
from `results/revision/*.csv` and `results/revision/summary.json`. Large per-record prediction files
(`results/oof_H*.csv`, `results/shap_X.csv`, `results/shap_values.npy`) and rendered figures are provided separately
in the full archive sent alongside this repository's creation (regenerate them locally with the scripts above).
