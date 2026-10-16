/*
 * CropWhisperAI - ESP32 field firmware with on-device heat-stress early warning (30 min ahead)
 * ---------------------------------------------------------------------------------------------
 * Every 5 min:  read DHT22, DS18B20, capacitive soil probe, BH1750, water-level strip
 *               -> validate each reading and set a status flag (same codes as the dataset)
 *               -> carry forward the last valid value of any sensor that failed
 *               -> push to the 24-h ring buffer and evaluate the logistic model (cropwhisper_model.h)
 *               -> time the prediction with micros()
 *               -> update the OLED, print one CSV line on Serial, upload JSON to the cloud endpoint
 *
 * Board: "ESP32 Dev Module" (arduino-esp32 2.0.x)
 * Libraries: DHT sensor library, Adafruit Unified Sensor, OneWire, DallasTemperature, BH1750 (claws),
 *            Adafruit SSD1306, Adafruit GFX, Adafruit BusIO
 *
 * Serial CSV columns (115200 baud):
 *   timestamp,location_id,air_temp_c,humidity_pct,soil_temp_c,soil_moisture_pct,light_lux,water_level_pct,
 *   sensor_status,risk_30min,warning,inference_us
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BH1750.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <time.h>
#include "cropwhisper_model.h"

// ------------------------------------------------------------------ configuration
#define LOCATION_ID      "VENGAL_SITE_01"
#define WIFI_SSID        "your-ssid"
#define WIFI_PASS        "your-password"
#define CLOUD_URL        ""              // REST endpoint accepting JSON POST; leave "" to use ThingSpeak
#define THINGSPEAK_KEY   ""              // ThingSpeak write API key; leave "" to disable upload
#define TZ_OFFSET_S      19800           // IST (UTC+05:30)

#define PIN_DHT          4               // DHT22 data (single-wire)
#define PIN_ONEWIRE      5               // DS18B20 data (1-Wire, 4.7 kOhm pull-up)
#define PIN_SOIL_ADC     34              // capacitive soil probe (ADC1, input-only pin)
#define PIN_WATER_ADC    35              // water-level strip (ADC1, input-only pin)
#define PIN_SDA          21              // I2C: BH1750 + SSD1306
#define PIN_SCL          22

// two-point calibration of the analogue sensors (raw 12-bit ADC counts) - measure for your probes
#define SOIL_RAW_DRY     3000            // probe in dry air
#define SOIL_RAW_WET     1300            // probe in water
#define WATER_RAW_EMPTY  0
#define WATER_RAW_FULL   2600

#define BH1750_MTREG     32              // default 69 saturates near 54.6 klx; 32 extends the range for full sunlight
const uint32_t PERIOD_MS = 5UL * 60UL * 1000UL;

// ------------------------------------------------------------------ devices
DHT dht(PIN_DHT, DHT22);
OneWire ow(PIN_ONEWIRE);
DallasTemperature ds(&ow);
BH1750 lightMeter(0x23);
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
bool oledOk = false, bhOk = false;

// last valid value of every sensor (carry-forward state)
float lastT = NAN, lastRH = NAN, lastTs = NAN, lastSM = NAN, lastLux = NAN, lastWL = NAN;
// v1.1 (revision): limit how long a carried-forward temperature may stand in for a failed sensor
#define CW_MAX_CARRY     3               // consecutive invalid readings (15 min) before warnings are suppressed
#define CW_WARMUP        13              // samples (65 min) needed before 60-min features are defined
static uint8_t badT = 0, badTs = 0;

static float mapClamp(int raw, int a, int b) {           // linear map a->0 %, b->100 %, clamped
  float v = 100.0f * (float)(raw - a) / (float)(b - a);
  return v < 0 ? 0 : (v > 100 ? 100 : v);
}

static bool timeNow(struct tm *ti) { return getLocalTime(ti, 50); }

static void connectWiFi(uint32_t timeoutMs) {
  if (WiFi.status() == WL_CONNECTED || strlen(WIFI_SSID) == 0) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < timeoutMs) delay(200);
}

// returns true if the record reached the cloud (or upload is disabled)
static bool upload(const char *ts, float t, float rh, float tsl, float sm, float lux, float wl,
                   const char *status, float p, bool warn) {
  if (strlen(CLOUD_URL) == 0 && strlen(THINGSPEAK_KEY) == 0) return true;
  connectWiFi(8000);
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  int code;
  if (strlen(CLOUD_URL) > 0) {
    char body[400];
    snprintf(body, sizeof body,
             "{\"timestamp\":\"%s\",\"location_id\":\"%s\",\"air_temperature\":%.1f,\"humidity\":%.1f,"
             "\"soil_temperature\":%.1f,\"soil_moisture\":%.1f,\"light\":%.0f,\"water_level\":%.1f,"
             "\"sensor_status\":\"%s\",\"risk_30min\":%.4f,\"warning\":%s}",
             ts, LOCATION_ID, t, rh, tsl, sm, lux, wl, status, p, warn ? "true" : "false");
    http.begin(CLOUD_URL);
    http.addHeader("Content-Type", "application/json");
    code = http.POST((uint8_t *)body, strlen(body));
  } else {
    char url[300];
    snprintf(url, sizeof url, "http://api.thingspeak.com/update?api_key=%s&field1=%.1f&field2=%.1f&field3=%.1f"
             "&field4=%.1f&field5=%.0f&field6=%.1f&field7=%.4f&field8=%d",
             THINGSPEAK_KEY, t, rh, tsl, sm, lux, wl, p, warn ? 1 : 0);
    http.begin(url);
    code = http.GET();
  }
  http.end();
  return code > 0 && code < 400;
}

static void showOled(float t, float rh, float tsl, float sm, float lux, float wl, float p, bool warn, const char *status) {
  if (!oledOk) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);  oled.print("CropWhisperAI");
  oled.setCursor(0, 11); oled.printf("Air %.1fC  RH %.0f%%", t, rh);
  oled.setCursor(0, 21); oled.printf("Soil %.1fC  SM %.0f%%", tsl, sm);
  oled.setCursor(0, 31); oled.printf("Lux %.0f  W %.0f%%", lux, wl);
  oled.setCursor(0, 41); oled.printf("Risk30 %.0f%%  %s", p * 100.0f, strcmp(status, "OK") ? "!" : "");
  if (warn) { oled.fillRect(0, 52, 128, 12, SSD1306_WHITE); oled.setTextColor(SSD1306_BLACK); }
  oled.setCursor(2, 54); oled.print(warn ? "HEAT WARNING <30 min" : "Normal");
  oled.display();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(PIN_SDA, PIN_SCL);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_SOIL_ADC, ADC_11db);
  analogSetPinAttenuation(PIN_WATER_ADC, ADC_11db);
  dht.begin();
  ds.begin();
  bhOk = lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE) && lightMeter.setMTreg(BH1750_MTREG);
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  connectWiFi(15000);
  configTime(TZ_OFFSET_S, 0, "pool.ntp.org", "time.google.com");
  Serial.println("timestamp,location_id,air_temp_c,humidity_pct,soil_temp_c,soil_moisture_pct,light_lux,"
                 "water_level_pct,sensor_status,risk_30min,warning,inference_us");
}

void loop() {
  static uint32_t last = 0;
  static bool first = true;
  if (!first && millis() - last < PERIOD_MS) { delay(50); return; }
  first = false;
  last = millis();

  // ---- read and validate; the first failing sensor sets the status flag
  const char *status = "OK";
  float t = dht.readTemperature(), rh = dht.readHumidity();
  if (isnan(t) || isnan(rh) || rh < 0 || rh > 100) { status = "DHT22_TIMEOUT"; t = lastT; rh = lastRH; if (badT < 255) badT++; }
  else { lastT = t; lastRH = rh; badT = 0; }

  ds.requestTemperatures();
  float tsl = ds.getTempCByIndex(0);
  if (tsl == DEVICE_DISCONNECTED_C || tsl < -10 || tsl > 85) { if (!strcmp(status, "OK")) status = "DS18B20_FAULT"; tsl = lastTs; if (badTs < 255) badTs++; }
  else { lastTs = tsl; badTs = 0; }

  int soilRaw = analogRead(PIN_SOIL_ADC);
  float sm = mapClamp(soilRaw, SOIL_RAW_DRY, SOIL_RAW_WET);
  if (soilRaw <= 0 || soilRaw >= 4095) { if (!strcmp(status, "OK")) status = "SOIL_PROBE_FAULT"; sm = lastSM; }
  else lastSM = sm;

  float lux = bhOk ? lightMeter.readLightLevel() : -1.0f;
  if (lux < 0) { if (!strcmp(status, "OK")) status = "BH1750_FAULT"; lux = lastLux; }
  else lastLux = lux;

  int wlRaw = analogRead(PIN_WATER_ADC);
  float wl = mapClamp(wlRaw, WATER_RAW_EMPTY, WATER_RAW_FULL);
  lastWL = wl;

  // ---- on-device prediction (same carry-forward inputs as the training pipeline)
  struct tm ti;
  bool haveTime = timeNow(&ti);
  float hour = haveTime ? ti.tm_hour + ti.tm_min / 60.0f : 12.0f;   // the deployed 8-feature model does not use hour
  cw_push(t, rh, tsl, lux);
  uint32_t t0 = micros();
  float margin = cw_margin(hour);
  uint32_t dtUs = micros() - t0;
  float p = 1.0f / (1.0f + expf(-margin));
  // v1.1: no warning during warm-up or while a temperature sensor has been carried forward for > 15 min;
  // a sustained fault is shown on the OLED (status flag) instead of being concealed by the carried value
  bool usable = (cw_n >= CW_WARMUP) && badT <= CW_MAX_CARRY && badTs <= CW_MAX_CARRY;
  bool warn = usable && !isnan(margin) && margin >= CW_ALARM_MARGIN;

  // ---- outputs
  char ts[24] = "1970-01-01T00:00:00";
  if (haveTime) strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &ti);
  showOled(t, rh, tsl, sm, lux, wl, p, warn, status);
  bool sent = upload(ts, t, rh, tsl, sm, lux, wl, status, p, warn);
  if (!sent && !strcmp(status, "OK")) status = "WIFI_RETRY";
  Serial.printf("%s,%s,%.1f,%.1f,%.1f,%.1f,%.0f,%.1f,%s,%.4f,%d,%lu\n",
                ts, LOCATION_ID, t, rh, tsl, sm, lux, wl, status, p, warn ? 1 : 0, (unsigned long)dtUs);
}
