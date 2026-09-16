/*
 * ═══════════════════════════════════════════════════════════════
 * ESP32 COMPLETE CRASH DETECTION SYSTEM - 3 COLOR LED VERSION
 * ═══════════════════════════════════════════════════════════════
 *
 * FULL INTEGRATED SYSTEM:
 *   ✅ MPU6050 - Crash detection (accelerometer + jerk)
 *   ✅ WiFi - Connection to NESSA 1863
 *   ✅ ML API - Severity classification (Normal/Moderate/Severe)
 *   ✅ 3-Color LED System - Severity indication
 *      🟢 GREEN (GPIO 26) = NORMAL
 *      🟡 YELLOW (GPIO 25) = MODERATE
 *      🔴 RED (GPIO 33) = SEVERE
 *   ✅ Buzzer (GPIO 32) - Synchronized audio alerts
 *   ✅ Serial Monitor - Real-time status
 *
 * COMPLETE FLOW:
 *   1. Monitor acceleration continuously @ 100Hz
 *   2. Detect crash (2.0g acceleration + 5.0g/s jerk)
 *   3. Keep sampling 2.5 s after the trigger so the 500-sample (5 s) window
 *      is CENTRED on the impact (250 pre + 250 post). The v2 API needs the
 *      whole 40-250 ms crash pulse inside the window; sending at the trigger
 *      instant put only ~10 ms of it at the window edge -> always Normal.
 *   4. Send all 6 fields (ax, ay, az, gx, gy, gz) to ML API via WiFi
 *   5. Get severity classification; if the API is unreachable, classify
 *      locally with the same crash-signature gate (classifyLocally)
 *   6. Trigger synchronized LED + buzzer alerts:
 *      - NORMAL: Green light + 1 beep
 *      - MODERATE: Yellow pulse + 3 beeps (yellow stays on)
 *      - SEVERE: Red rapid flash + 5 beeps (red stays on - EMERGENCY)
 *   7. Store the incident (raw window + both classifications + profile) at
 *      /api/v1/events; if that fails the full event is printed as an
 *      [UNSENT_EVENT] line for tools/ingest_serial_log.py
 *
 * THRESHOLD PROFILE (compile-time):
 *   Default build = production (the VZCrash-derived thesis taxonomy).
 *   Uncomment THRESHOLD_PROFILE_DEMO_RC below for the scaled RC-car
 *   demonstration profile. Thresholds come from threshold_profiles.h, which
 *   is generated from the server's artifacts/threshold_profiles.json, so the
 *   device and the API apply the same gate. The banner and every
 *   classification line print the active profile.
 *
 * DATA COLLECTION (serial commands, 115200 baud, newline-terminated):
 *   help | status | collect on|off | label normal|impact_low|impact_high
 *   run speed=<km/h> barrier=<none|padded|rigid> angle=<deg> crush=<none|foam|lattice|spring> cond=<id> notes=<text>
 *   capture   - record a window centred on now (for Normal manoeuvres)
 *   panic     - store a manual_panic event
 *
 * WIRING:
 *   MPU6050:     GPIO 21 (SDA), GPIO 22 (SCL)
 *   RED LED:     GPIO 33 + 220Ω resistor
 *   YELLOW LED:  GPIO 25 + 220Ω resistor
 *   GREEN LED:   GPIO 26 + 220Ω resistor
 *   BUZZER:      GPIO 32
 *
 * ═══════════════════════════════════════════════════════════════
 */

// ═══════════════════════════════════════════════════════════════
// THRESHOLD PROFILE — leave commented for production
// ═══════════════════════════════════════════════════════════════
// #define THRESHOLD_PROFILE_DEMO_RC

#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include "threshold_profiles.h"
#include "signature_gate.h"
#include "secrets.h"         // WIFI_SSID, WIFI_PASSWORD, EVENTS_API_KEY (copy secrets.example.h)

// ═══════════════════════════════════════════════════════════════
// GPIO CONFIGURATION
// ═══════════════════════════════════════════════════════════════

// 3-Color LED System
#define LED_RED 33           // GPIO 33: Red LED (SEVERE/Emergency)
#define LED_YELLOW 25        // GPIO 25: Yellow LED (MODERATE/Warning)
#define LED_GREEN 26         // GPIO 26: Green LED (NORMAL/Safe)
#define BUZZER_PIN 32        // GPIO 32: Audio alert buzzer

// I2C for MPU6050
#define I2C_SDA 21           // GPIO 21: MPU6050 SDA
#define I2C_SCL 22           // GPIO 22: MPU6050 SCL

// #define GPS_TX      17

// ═══════════════════════════════════════════════════════════════
// ML API CONFIGURATION
// ═══════════════════════════════════════════════════════════════

const char* ML_API_URL = "https://accident-severity-api-production.up.railway.app/predict";
const char* EVENTS_API_URL = "https://accident-severity-api-production.up.railway.app/api/v1/events";
const uint16_t ML_API_TIMEOUT_MS = 15000;

// ═══════════════════════════════════════════════════════════════
// IMU HARDWARE
// ═══════════════════════════════════════════════════════════════

#define MPU6050_ADDR 0x68
#define SENSOR_INTERVAL_MS 10
#define PRINT_INTERVAL_MS 100

// ═══════════════════════════════════════════════════════════════
// CRASH DETECTION PARAMETERS
// ═══════════════════════════════════════════════════════════════

#define BUFFER_SIZE 500           // 500 samples = 5 seconds @ 100Hz
#define ACCEL_THRESHOLD_G 2.0f    // 2.0g acceleration magnitude
#define JERK_THRESHOLD_G_PER_SEC 5.0f  // 5.0g/s jerk rate
#define POST_TRIGGER_SAMPLES 250  // 2.5 s captured after the trigger (impact centred)

// mpu.initialize() defaults to ±2 g, which clips real crashes (2-7 g). setup()
// selects ±16 g explicitly, so the divisor is 2048 LSB/g (not 16384).
#define RAW_ACCEL_TO_G (1.0f / 2048.0f)    // MPU6050 ±16g range
// ±500 dps: 2.3% of verified VZCrash crashes exceed 250 dps (0.6% exceed 500).
#define RAW_GYRO_TO_DPS (1.0f / 65.5f)     // MPU6050 ±500 dps range

// ═══════════════════════════════════════════════════════════════
// SEVERITY LEVELS
// ═══════════════════════════════════════════════════════════════

enum SeverityLevel {
  SEVERITY_NORMAL = 0,      // 🟢 Safe
  SEVERITY_MODERATE = 1,    // 🟡 Warning
  SEVERITY_SEVERE = 2       // 🔴 Emergency
};

const char* SEVERITY_NAMES[] = {"Normal", "Moderate", "Severe"};

// What started the current capture
enum CaptureKind {
  CAPTURE_TRIGGER = 0,      // accel + jerk threshold
  CAPTURE_MANUAL = 1        // "capture" serial command (Normal manoeuvres)
};

// ═══════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════

struct IMUSample {
  float ax_g;      // Accelerometer X in g-units
  float ay_g;      // Accelerometer Y in g-units
  float az_g;      // Accelerometer Z in g-units
  float gx_dps;    // Gyroscope X in degrees/sec
  float gy_dps;    // Gyroscope Y in degrees/sec
  float gz_dps;    // Gyroscope Z in degrees/sec
  float accel_mag; // Magnitude for crash detection
  uint32_t timestamp_ms;
};

struct CrashResult {
  bool detected;
  int severity_class;
  const char* severity_name;
  float confidence;
  float peak_magnitude_g;
  const char* source;          // "api" or "local"
  char server_profile[32];     // profile the API classified under ("" for local)
};

IMUSample imu_buffer[BUFFER_SIZE];
uint16_t buffer_index = 0;
bool buffer_full = false;

// Window frozen at classification time, oldest sample first. /predict,
// classifyLocally() and /api/v1/events all use exactly these samples.
float win_ax[BUFFER_SIZE], win_ay[BUFFER_SIZE], win_az[BUFFER_SIZE];
float win_gx[BUFFER_SIZE], win_gy[BUFFER_SIZE], win_gz[BUFFER_SIZE];
float raw_peak_g = 0;      // peak of the transmitted array, before the API's filter
int raw_peak_index = 0;

// ═══════════════════════════════════════════════════════════════
// GLOBAL VARIABLES
// ═══════════════════════════════════════════════════════════════

MPU6050 mpu(MPU6050_ADDR);
bool mpu_ready = false;
bool wifi_ready = false;

int16_t accelX, accelY, accelZ;
int16_t gyroX, gyroY, gyroZ;
float accel_x_g = 0, accel_y_g = 0, accel_z_g = 0;
float prev_accel_mag_g = 0;

unsigned long last_sensor_read = 0;
unsigned long last_print = 0;
unsigned long last_crash_detection = 0;
uint32_t sample_count = 0;
uint32_t crash_count = 0;

bool crash_detected_flag = false;   // a capture is in progress
CaptureKind capture_kind = CAPTURE_TRIGGER;
uint16_t post_trigger_remaining = 0;
float trigger_mag_g = 0;  // magnitude at trigger, for the [PEAKS] alignment log
bool api_request_in_progress = false;
CrashResult last_crash_result = {false, 0, "None", 0.0, 0.0, "none", ""};
GateResult last_local_gate = {0, 0, 0, false, 0};

// RC-car data collection state (set over serial)
bool collect_mode = false;
char collect_label[16] = "";        // normal | impact_low | impact_high; cleared after each event
char run_speed_kmh[12] = "";
char run_barrier[16] = "";
char run_angle_deg[12] = "";
char run_crush[16] = "";
char run_cond[24] = "";
char run_notes[64] = "";
char serial_line[160];
uint8_t serial_len = 0;

// ═══════════════════════════════════════════════════════════════
// MOUNT ORIENTATION CHECK
// ═══════════════════════════════════════════════════════════════
// The ML model uses per-axis features (ax/ay/az) learned from data with gravity
// on +Z (VZCrash median az = 1.0 g). A board mounted on its side puts gravity on
// X or Y and silently skews every prediction, so verify at boot (vehicle still).

void checkMountOrientation() {
  const int N = 100;
  float sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < N; i++) {
    mpu.getAcceleration(&accelX, &accelY, &accelZ);
    sx += accelX * RAW_ACCEL_TO_G;
    sy += accelY * RAW_ACCEL_TO_G;
    sz += accelZ * RAW_ACCEL_TO_G;
    delay(SENSOR_INTERVAL_MS);
  }
  sx /= N; sy /= N; sz /= N;
  Serial.printf("[MOUNT] At rest: ax=%.2fg ay=%.2fg az=%.2fg\n", sx, sy, sz);
  if (sz < 0.8f || fabs(sx) > 0.35f || fabs(sy) > 0.35f) {
    Serial.println(F("[MOUNT] WARNING: gravity is not on +Z. Mount the MPU6050 flat, Z axis up,"));
    Serial.println(F("[MOUNT]          or severity predictions will be unreliable."));
    for (int i = 0; i < 5; i++) {
      digitalWrite(LED_RED, HIGH); digitalWrite(LED_YELLOW, HIGH);
      delay(150);
      digitalWrite(LED_RED, LOW); digitalWrite(LED_YELLOW, LOW);
      delay(150);
    }
  } else {
    Serial.println(F("[MOUNT] ✓ Orientation OK (Z axis up)"));
  }
}

// ═══════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize all LED pins
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Start with all OFF
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  printBanner();

  // ─────────────────────────────────────────────────────────────
  // Initialize I2C & MPU6050
  // ─────────────────────────────────────────────────────────────
  Serial.println(F("[I2C] Initializing..."));
  Wire.begin(I2C_SDA, I2C_SCL, 400000);
  delay(100);

  Serial.println(F("[MPU6050] Initializing sensor..."));
  mpu.initialize();
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_16);   // must match RAW_ACCEL_TO_G
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);    // must match RAW_GYRO_TO_DPS

  if (!mpu.testConnection()) {
    Serial.println(F("[ERROR] MPU6050 not found!"));
    mpu_ready = false;
    // Red error indication
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_RED, HIGH);
      delay(200);
      digitalWrite(LED_RED, LOW);
      delay(200);
    }
  } else {
    Serial.println(F("[MPU6050] ✓ Connected! (Accel + Gyro)"));
    mpu_ready = true;
    checkMountOrientation();
    // Green success indication
    digitalWrite(LED_GREEN, HIGH);
    delay(500);
    digitalWrite(LED_GREEN, LOW);
  }

  // ─────────────────────────────────────────────────────────────
  // Initialize WiFi
  // ─────────────────────────────────────────────────────────────
  Serial.println();
  Serial.print(F("[WiFi] Connecting to: "));
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int wifi_attempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempts < 20) {
    delay(500);
    Serial.print(F("."));
    wifi_attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(F("[WiFi] ✓ Connected!"));
    Serial.print(F("[WiFi] IP: "));
    Serial.println(WiFi.localIP());
    wifi_ready = true;
    // Yellow success indication
    digitalWrite(LED_YELLOW, HIGH);
    delay(300);
    digitalWrite(LED_YELLOW, LOW);
  } else {
    Serial.println(F("[WiFi] ✗ Connection failed!"));
    wifi_ready = false;
  }

  Serial.println();
  Serial.println(F("═══════════════════════════════════════════════════════════"));
  Serial.println(F("Time(ms) | Accel(g) | Jerk(g/s) | WiFi | MPU | Status"));
  Serial.println(F("═══════════════════════════════════════════════════════════"));

  last_sensor_read = millis();
  last_print = millis();
}

// ═══════════════════════════════════════════════════════════════
// MAIN LOOP
// ═══════════════════════════════════════════════════════════════

void loop() {
  unsigned long now = millis();

  // Check WiFi status
  if (WiFi.status() != WL_CONNECTED && wifi_ready) {
    wifi_ready = false;
  }

  handleSerialCommands();

  // ─────────────────────────────────────────────────────────────
  // Read IMU Sensors at 100Hz (every 10ms)
  // ─────────────────────────────────────────────────────────────
  if (now - last_sensor_read >= SENSOR_INTERVAL_MS) {
    if (mpu_ready) {
      // Read accelerometer and gyroscope
      mpu.getAcceleration(&accelX, &accelY, &accelZ);
      mpu.getRotation(&gyroX, &gyroY, &gyroZ);

      // Convert raw values to standard units
      accel_x_g = accelX * RAW_ACCEL_TO_G;
      accel_y_g = accelY * RAW_ACCEL_TO_G;
      float accel_z_g = accelZ * RAW_ACCEL_TO_G;

      float gyro_x_dps = gyroX * RAW_GYRO_TO_DPS;
      float gyro_y_dps = gyroY * RAW_GYRO_TO_DPS;
      float gyro_z_dps = gyroZ * RAW_GYRO_TO_DPS;

      // Calculate acceleration magnitude
      float mag = sqrt(accel_x_g * accel_x_g +
                      accel_y_g * accel_y_g +
                      accel_z_g * accel_z_g);

      // Calculate jerk (rate of change of acceleration)
      float jerk_g_per_sec = (mag - prev_accel_mag_g) / (SENSOR_INTERVAL_MS / 1000.0f);

      // ─────────────────────────────────────────────────────────
      // CRASH DETECTION: Dual Threshold
      // ─────────────────────────────────────────────────────────
      bool accel_threshold = (mag >= ACCEL_THRESHOLD_G);
      bool jerk_threshold = (fabs(jerk_g_per_sec) >= JERK_THRESHOLD_G_PER_SEC);

      if (accel_threshold && jerk_threshold && !crash_detected_flag) {
        if (now - last_crash_detection > 2000) {  // Debounce: min 2 sec between crashes
          crash_detected_flag = true;
          capture_kind = CAPTURE_TRIGGER;
          last_crash_detection = now;

          Serial.print(F("[CRASH] Detected at "));
          Serial.print(now);
          Serial.print(F(" ms | Accel="));
          Serial.print(mag, 2);
          Serial.print(F("g, Jerk="));
          Serial.print(jerk_g_per_sec, 2);
          Serial.println(F("g/s"));
          Serial.println(F("[CRASH] Capturing 2.5 s post-impact data..."));

          // Quick alert: all LEDs ON. No delay() here — blocking would drop the
          // samples right after the impact, which the API needs.
          digitalWrite(LED_RED, HIGH);
          digitalWrite(LED_YELLOW, HIGH);
          digitalWrite(LED_GREEN, HIGH);
          post_trigger_remaining = POST_TRIGGER_SAMPLES;
          trigger_mag_g = mag;
        }
      }

      // Store sample in circular buffer
      imu_buffer[buffer_index].ax_g = accel_x_g;
      imu_buffer[buffer_index].ay_g = accel_y_g;
      imu_buffer[buffer_index].az_g = accel_z_g;
      imu_buffer[buffer_index].gx_dps = gyro_x_dps;
      imu_buffer[buffer_index].gy_dps = gyro_y_dps;
      imu_buffer[buffer_index].gz_dps = gyro_z_dps;
      imu_buffer[buffer_index].accel_mag = mag;
      imu_buffer[buffer_index].timestamp_ms = now;

      buffer_index++;
      if (buffer_index >= BUFFER_SIZE) {
        buffer_index = 0;
        buffer_full = true;
      }

      if (crash_detected_flag && post_trigger_remaining > 0) {
        post_trigger_remaining--;
      }

      prev_accel_mag_g = mag;
      sample_count++;
    }
    last_sensor_read = now;
  }

  // ─────────────────────────────────────────────────────────────
  // Classify once the post-impact half of the window is captured
  // (no WiFi -> sendCrashToMLAPI fails and classifyLocally decides)
  // ─────────────────────────────────────────────────────────────
  if (crash_detected_flag && post_trigger_remaining == 0 && buffer_full && !api_request_in_progress) {
    api_request_in_progress = true;
    Serial.println(F("[ML API] Collecting and sending crash data..."));
    processCrash();
    crash_detected_flag = false;
    api_request_in_progress = false;
  }

  // ─────────────────────────────────────────────────────────────
  // Print status every 100ms (suppressed in collect mode so commands stay readable)
  // ─────────────────────────────────────────────────────────────
  if (now - last_print >= PRINT_INTERVAL_MS) {
    if (mpu_ready && buffer_full && sample_count > 1 && !collect_mode) {
      uint16_t curr_idx = (buffer_index - 1 + BUFFER_SIZE) % BUFFER_SIZE;
      uint16_t prev_idx = (buffer_index - 2 + BUFFER_SIZE) % BUFFER_SIZE;

      float time_delta = (imu_buffer[curr_idx].timestamp_ms -
                         imu_buffer[prev_idx].timestamp_ms) / 1000.0f;
      float jerk = 0;
      if (time_delta > 0) {
        jerk = (imu_buffer[curr_idx].accel_mag -
                imu_buffer[prev_idx].accel_mag) / time_delta;
      }

      Serial.print(now);
      Serial.print(F("   | "));
      Serial.print(imu_buffer[curr_idx].accel_mag, 2);
      Serial.print(F("g    | "));
      Serial.print(jerk, 2);
      Serial.print(F("    | "));
      Serial.print(wifi_ready ? F("OK") : F("--"));
      Serial.print(F("   | "));
      Serial.print(mpu_ready ? F("OK") : F("--"));
      Serial.print(F("  | "));
      Serial.println(F("RUNNING"));
    }
    last_print = now;
  }
}

// ═══════════════════════════════════════════════════════════════
// CRASH PROCESSING
// ═══════════════════════════════════════════════════════════════

// Copy the circular buffer, oldest first, into the win_* arrays.
void freezeWindow() {
  raw_peak_g = 0;
  raw_peak_index = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    uint16_t idx = (buffer_index + i) % BUFFER_SIZE;
    win_ax[i] = imu_buffer[idx].ax_g;
    win_ay[i] = imu_buffer[idx].ay_g;
    win_az[i] = imu_buffer[idx].az_g;
    win_gx[i] = imu_buffer[idx].gx_dps;
    win_gy[i] = imu_buffer[idx].gy_dps;
    win_gz[i] = imu_buffer[idx].gz_dps;
    if (imu_buffer[idx].accel_mag > raw_peak_g) {
      raw_peak_g = imu_buffer[idx].accel_mag;
      raw_peak_index = i;
    }
  }
}

void processCrash() {
  freezeWindow();

  // Send to ML API; fall back to the on-device gate
  bool api_success = sendCrashToMLAPI();
  if (!api_success) {
    Serial.println(F("[ALERT] API unavailable - classifying locally"));
    classifyLocally();
  }

  printClassificationLine();
  triggerSeverityAlert(last_crash_result.severity_class);
  storeEvent(collect_mode ? "rc_collection" : "trigger", true);
  if (collect_mode) {
    collect_label[0] = '\0';   // force a deliberate label for the next run
    Serial.println(F("[COLLECT] Label cleared. Set 'label ...' before the next run."));
  }
}

// ═══════════════════════════════════════════════════════════════
// LOCAL CLASSIFICATION (API unreachable)
// ═══════════════════════════════════════════════════════════════
// Same filter and gate as the server (signature_gate.h, thresholds from
// threshold_profiles.h). No model on the device, so a matching signature is
// graded by impulse alone: Severe if impulse >= severe_impulse_gs.

void classifyLocally() {
  last_local_gate = evaluateSignature(win_ax, win_ay, win_az, ACTIVE_THRESHOLD_PROFILE);
  crash_count++;
  last_crash_result.detected = true;
  last_crash_result.severity_class = last_local_gate.severity_class;
  last_crash_result.severity_name = SEVERITY_NAMES[last_local_gate.severity_class];
  last_crash_result.confidence = 0;
  last_crash_result.peak_magnitude_g = last_local_gate.peak_g;
  last_crash_result.source = "local";
  last_crash_result.server_profile[0] = '\0';
  displayCrashResults();
}

// One line per classification, always naming the profile that produced it.
void printClassificationLine() {
  Serial.printf("[CLASSIFY] source=%s profile=%s class=%d %s peak=%.2fg",
                last_crash_result.source, ACTIVE_THRESHOLD_PROFILE.name,
                last_crash_result.severity_class, last_crash_result.severity_name,
                last_crash_result.peak_magnitude_g);
  if (strcmp(last_crash_result.source, "api") == 0) {
    Serial.printf(" server_profile=%s\n", last_crash_result.server_profile);
    if (strcmp(last_crash_result.server_profile, ACTIVE_THRESHOLD_PROFILE.name) != 0) {
      Serial.printf("[PROFILE] MISMATCH: device built with '%s' but API classified under '%s'.\n",
                    ACTIVE_THRESHOLD_PROFILE.name, last_crash_result.server_profile);
      Serial.println(F("[PROFILE] Rebuild the firmware or set THRESHOLD_PROFILE on the server."));
    }
  } else {
    Serial.printf(" excursion=%.0fms impulse=%.3fg.s signature=%s\n",
                  last_local_gate.excursion_ms, last_local_gate.impulse_gs,
                  last_local_gate.signature_match ? "yes" : "no");
  }
}

// ═══════════════════════════════════════════════════════════════
// SEVERITY-BASED LED & BUZZER ALERTS
// ═══════════════════════════════════════════════════════════════

void triggerSeverityAlert(int severity_class) {
  // Turn off all LEDs first
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_GREEN, LOW);
  noTone(BUZZER_PIN);

  switch (severity_class) {
    case SEVERITY_NORMAL:
      // 🟢 GREEN: Safe, no alert needed
      Serial.println(F("🟢 SEVERITY: NORMAL - No alert needed"));
      digitalWrite(LED_GREEN, HIGH);  // Green LED ON continuously
      tone(BUZZER_PIN, 1000, 100);   // Single beep
      delay(150);
      noTone(BUZZER_PIN);
      break;

    case SEVERITY_MODERATE:
      // 🟡 YELLOW: Warning, medium alert
      Serial.println(F("🟡 SEVERITY: MODERATE - Warning alert"));
      // Pulse pattern: 3 pulses synchronized
      for (int i = 0; i < 3; i++) {
        digitalWrite(LED_YELLOW, HIGH);
        tone(BUZZER_PIN, 1500, 150);
        delay(300);
        digitalWrite(LED_YELLOW, LOW);
        noTone(BUZZER_PIN);
        delay(200);
      }
      // Keep yellow on after alert
      digitalWrite(LED_YELLOW, HIGH);
      break;

    case SEVERITY_SEVERE:
      // 🔴 RED: Emergency, rapid alert
      Serial.println(F("🔴 SEVERITY: SEVERE - EMERGENCY ALERT!!!"));
      // Rapid pulse pattern: 5 rapid flashes
      for (int i = 0; i < 5; i++) {
        digitalWrite(LED_RED, HIGH);
        tone(BUZZER_PIN, 2000, 100);  // High-pitched beep
        delay(150);
        digitalWrite(LED_RED, LOW);
        noTone(BUZZER_PIN);
        delay(150);
      }
      // Keep RED LED ON continuously (EMERGENCY mode)
      digitalWrite(LED_RED, HIGH);
      break;
  }
}

// ═══════════════════════════════════════════════════════════════
// BUZZER HELPER
// ═══════════════════════════════════════════════════════════════

void alertBuzzer(int beeps, int delay_ms) {
  for (int i = 0; i < beeps; i++) {
    tone(BUZZER_PIN, 1000, delay_ms / 2);
    delay(delay_ms);
  }
  noTone(BUZZER_PIN);
}

// ═══════════════════════════════════════════════════════════════
// ML API COMMUNICATION
// ═══════════════════════════════════════════════════════════════

void addWindowToJson(JsonDocument& doc) {
  JsonArray ax_array = doc["ax"].to<JsonArray>();
  JsonArray ay_array = doc["ay"].to<JsonArray>();
  JsonArray az_array = doc["az"].to<JsonArray>();
  JsonArray gx_array = doc["gx"].to<JsonArray>();
  JsonArray gy_array = doc["gy"].to<JsonArray>();
  JsonArray gz_array = doc["gz"].to<JsonArray>();
  for (int i = 0; i < BUFFER_SIZE; i++) {
    ax_array.add(win_ax[i]);
    ay_array.add(win_ay[i]);
    az_array.add(win_az[i]);
    gx_array.add(win_gx[i]);
    gy_array.add(win_gy[i]);
    gz_array.add(win_gz[i]);
  }
}

bool sendCrashToMLAPI() {
  if (!WiFi.isConnected()) {
    Serial.println(F("[ERROR] WiFi not connected!"));
    return false;
  }

  // JSON payload with all 6 IMU fields (request contract unchanged)
  JsonDocument doc;
  addWindowToJson(doc);

  String payload;
  serializeJson(doc, payload);

  Serial.print(F("[ML API] Payload size: "));
  Serial.print(payload.length());
  Serial.println(F(" bytes"));
  Serial.println(F("[ML API] Sending POST request..."));

  HTTPClient http;
  http.setTimeout(ML_API_TIMEOUT_MS);
  http.begin(ML_API_URL);
  http.addHeader("Content-Type", "application/json");

  unsigned long send_start = millis();
  int httpResponseCode = http.POST(payload);
  unsigned long send_time = millis() - send_start;

  if (httpResponseCode == 200) {
    Serial.print(F("[ML API] Response code: "));
    Serial.println(httpResponseCode);
    Serial.print(F("[ML API] Response time: "));
    Serial.print(send_time);
    Serial.println(F(" ms"));

    String response = http.getString();
    JsonDocument response_doc;
    DeserializationError error = deserializeJson(response_doc, response);

    if (!error) {
      crash_count++;

      int cls = response_doc["severity_class"] | 0;
      last_crash_result.detected = true;
      last_crash_result.severity_class = cls;
      last_crash_result.severity_name = SEVERITY_NAMES[constrain(cls, 0, 2)];
      last_crash_result.confidence = response_doc["confidence"];
      last_crash_result.peak_magnitude_g = response_doc["crash_signature"]["peak_g"];
      last_crash_result.source = "api";
      strlcpy(last_crash_result.server_profile, response_doc["profile"] | "unreported",
              sizeof(last_crash_result.server_profile));

      displayCrashResults();
      Serial.print(F("[ML API] P(crash)="));
      Serial.print((float)response_doc["p_crash"], 3);
      Serial.print(F("  pulse above floor="));
      Serial.print((float)response_doc["crash_signature"]["excursion_ms"], 0);
      Serial.print(F(" ms  decided by: "));
      Serial.println((const char*)response_doc["label_source"]);
      // device vs raw array: window alignment (trigger sits at index 250).
      // raw array vs API: the API's 20 Hz low-pass removing a pulse shorter than ~30 ms.
      Serial.printf("[PEAKS] device=%.2fg  raw_array=%.2fg @idx %d  api=%.2fg  excursion=%.0fms\n",
                    trigger_mag_g, raw_peak_g, raw_peak_index,
                    last_crash_result.peak_magnitude_g,
                    (float)response_doc["crash_signature"]["excursion_ms"]);
      http.end();
      return true;
    }
  } else {
    Serial.print(F("[ERROR] HTTP POST failed. Code: "));
    Serial.println(httpResponseCode);
  }

  http.end();
  return false;
}

// ═══════════════════════════════════════════════════════════════
// INCIDENT STORE (/api/v1/events)
// ═══════════════════════════════════════════════════════════════
// Every incident keeps its raw window. If the POST fails, the complete event
// is printed as one [UNSENT_EVENT] line so a captured serial log still holds it
// (tools/ingest_serial_log.py uploads those lines later).

void storeEvent(const char* event_type, bool classified) {
  JsonDocument doc;
  doc["event_type"] = event_type;
  doc["device_id"] = WiFi.macAddress();
  doc["device_profile"] = ACTIVE_THRESHOLD_PROFILE.name;
  if (classified) {
    doc["classification_source"] = last_crash_result.source;
    doc["device_severity_class"] = last_crash_result.severity_class;
  } else {
    doc["classification_source"] = "none";
  }
  JsonObject metrics = doc["device_metrics"].to<JsonObject>();
  metrics["capture"] = capture_kind == CAPTURE_TRIGGER ? "trigger" : "manual";
  metrics["trigger_mag_g"] = trigger_mag_g;
  metrics["raw_peak_g"] = raw_peak_g;
  metrics["raw_peak_index"] = raw_peak_index;
  metrics["profile_requested"] = ACTIVE_THRESHOLD_PROFILE.requested;
  if (strcmp(event_type, "rc_collection") == 0) {
    if (collect_label[0]) doc["ground_truth"] = collect_label;
    JsonObject run = doc["run_conditions"].to<JsonObject>();
    run["speed_kmh"] = run_speed_kmh;
    run["barrier"] = run_barrier;
    run["angle_deg"] = run_angle_deg;
    run["crush"] = run_crush;
    run["cond"] = run_cond;
    run["notes"] = run_notes;
  }
  addWindowToJson(doc);

  String payload;
  serializeJson(doc, payload);

  int code = -1;
  if (WiFi.isConnected()) {
    HTTPClient http;
    http.setTimeout(ML_API_TIMEOUT_MS);
    http.begin(EVENTS_API_URL);
    http.addHeader("Content-Type", "application/json");
    if (strlen(EVENTS_API_KEY) > 0) http.addHeader("X-API-Key", EVENTS_API_KEY);
    code = http.POST(payload);
    if (code == 201) {
      JsonDocument resp;
      if (!deserializeJson(resp, http.getString())) {
        Serial.printf("[EVENT] stored id=%d type=%s device_profile=%s server_profile=%s excluded_from_performance=%s\n",
                      (int)(resp["id"] | -1), event_type, ACTIVE_THRESHOLD_PROFILE.name,
                      (const char*)(resp["server_profile"] | "?"),
                      (const char*)(resp["excluded_from_performance"] | "no"));
      }
    } else {
      Serial.printf("[EVENT] store failed, HTTP %d: %s\n", code, http.getString().c_str());
    }
    http.end();
  }
  if (code != 201) {
    Serial.print(F("[UNSENT_EVENT] "));
    Serial.println(payload);
  }
}

// ═══════════════════════════════════════════════════════════════
// SERIAL COMMANDS (data collection)
// ═══════════════════════════════════════════════════════════════

bool runConditionsComplete() {
  return run_speed_kmh[0] && run_barrier[0] && run_angle_deg[0] && run_crush[0];
}

void copyField(char* dst, size_t n, const char* value) {
  strlcpy(dst, value, n);
}

void parseRunConditions(char* args) {
  for (char* tok = strtok(args, " "); tok; tok = strtok(nullptr, " ")) {
    char* eq = strchr(tok, '=');
    if (!eq) continue;
    *eq = '\0';
    const char* v = eq + 1;
    if (!strcmp(tok, "speed")) copyField(run_speed_kmh, sizeof(run_speed_kmh), v);
    else if (!strcmp(tok, "barrier")) copyField(run_barrier, sizeof(run_barrier), v);
    else if (!strcmp(tok, "angle")) copyField(run_angle_deg, sizeof(run_angle_deg), v);
    else if (!strcmp(tok, "crush")) copyField(run_crush, sizeof(run_crush), v);
    else if (!strcmp(tok, "cond")) copyField(run_cond, sizeof(run_cond), v);
    else if (!strcmp(tok, "notes")) copyField(run_notes, sizeof(run_notes), v);
    else Serial.printf("[COLLECT] unknown run field '%s'\n", tok);
  }
}

void printCollectStatus() {
  Serial.printf("[COLLECT] mode=%s label=%s speed=%s km/h barrier=%s angle=%s deg crush=%s cond=%s notes=%s profile=%s\n",
                collect_mode ? "on" : "off", collect_label[0] ? collect_label : "(unset)",
                run_speed_kmh, run_barrier, run_angle_deg, run_crush, run_cond, run_notes,
                ACTIVE_THRESHOLD_PROFILE.name);
}

void handleCommand(char* line) {
  char* cmd = strtok(line, " ");
  if (!cmd) return;
  char* rest = strtok(nullptr, "");
  if (!strcmp(cmd, "help")) {
    Serial.println(F("[COLLECT] commands: status | collect on|off | label normal|impact_low|impact_high |"));
    Serial.println(F("[COLLECT]   run speed=<km/h> barrier=<none|padded|rigid> angle=<deg> crush=<none|foam|lattice|spring> cond=<id> notes=<text> |"));
    Serial.println(F("[COLLECT]   capture | panic"));
  } else if (!strcmp(cmd, "status")) {
    printCollectStatus();
  } else if (!strcmp(cmd, "collect")) {
    if (rest && !strcmp(rest, "on")) {
      if (!runConditionsComplete()) {
        Serial.println(F("[COLLECT] set run conditions first: run speed=.. barrier=.. angle=.. crush=.."));
        return;
      }
      collect_mode = true;
    } else {
      collect_mode = false;
    }
    printCollectStatus();
  } else if (!strcmp(cmd, "label")) {
    if (rest && (!strcmp(rest, "normal") || !strcmp(rest, "impact_low") || !strcmp(rest, "impact_high"))) {
      copyField(collect_label, sizeof(collect_label), rest);
    } else {
      Serial.println(F("[COLLECT] label must be normal, impact_low or impact_high"));
    }
    printCollectStatus();
  } else if (!strcmp(cmd, "run")) {
    if (rest) parseRunConditions(rest);
    printCollectStatus();
  } else if (!strcmp(cmd, "capture")) {
    if (!collect_mode) {
      Serial.println(F("[COLLECT] capture is only available in collect mode"));
    } else if (crash_detected_flag || !buffer_full) {
      Serial.println(F("[COLLECT] busy or buffer not yet full; try again"));
    } else {
      crash_detected_flag = true;
      capture_kind = CAPTURE_MANUAL;
      post_trigger_remaining = POST_TRIGGER_SAMPLES;
      trigger_mag_g = prev_accel_mag_g;
      last_crash_detection = millis();
      Serial.println(F("[COLLECT] capturing window centred on now (2.5 s)..."));
    }
  } else if (!strcmp(cmd, "panic")) {
    if (!buffer_full) {
      Serial.println(F("[PANIC] buffer not yet full"));
      return;
    }
    freezeWindow();
    capture_kind = CAPTURE_MANUAL;
    Serial.println(F("[PANIC] manual panic event"));
    storeEvent("manual_panic", false);
  } else {
    Serial.printf("[COLLECT] unknown command '%s' (type help)\n", cmd);
  }
}

void handleSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      serial_line[serial_len] = '\0';
      if (serial_len) handleCommand(serial_line);
      serial_len = 0;
    } else if (serial_len < sizeof(serial_line) - 1) {
      serial_line[serial_len++] = c;
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// DISPLAY CRASH RESULTS
// ═══════════════════════════════════════════════════════════════

void displayCrashResults() {
  Serial.println(F(""));
  Serial.println(F("╔════════════════════════════════════════════════════════╗"));
  Serial.println(F("║         CRASH ANALYSIS COMPLETE                       ║"));
  Serial.println(F("╠════════════════════════════════════════════════════════╣"));
  Serial.print(F("║ Incident #"));
  Serial.print(crash_count);
  Serial.println(F("                                          ║"));
  Serial.print(F("║ Severity Class: "));
  Serial.print(last_crash_result.severity_class);
  Serial.println(F("                                   ║"));
  Serial.print(F("║ Severity Name: "));
  Serial.print(last_crash_result.severity_name);
  Serial.println(F("                         ║"));
  Serial.print(F("║ Confidence: "));
  Serial.print(last_crash_result.confidence * 100, 1);
  Serial.println(F("%                              ║"));
  Serial.print(F("║ Peak Acceleration: "));
  Serial.print(last_crash_result.peak_magnitude_g, 2);
  Serial.println(F("g                     ║"));
  Serial.print(F("║ Decided by: "));
  Serial.print(last_crash_result.source);
  Serial.print(F("   Profile: "));
  Serial.println(ACTIVE_THRESHOLD_PROFILE.name);
  Serial.println(F("╚════════════════════════════════════════════════════════╝"));
  Serial.println(F(""));
}

// ═══════════════════════════════════════════════════════════════
// BANNER
// ═══════════════════════════════════════════════════════════════

void printProfileBanner() {
  const ThresholdProfile& p = ACTIVE_THRESHOLD_PROFILE;
  Serial.println(F("╠════════════════════════════════════════════════════════╣"));
  Serial.printf("║   THRESHOLD PROFILE: %s\n", p.name);
  Serial.printf("║     requested by build: %s\n", p.requested);
  if (p.note) Serial.printf("║     NOTE: %s\n", p.note);
  Serial.printf("║     peak band:       %.2f g <= peak < %.2f g\n", p.peak_min_g, p.peak_max_g);
  Serial.printf("║     pulse duration:  %.0f-%.0f ms continuously >= %.2f g\n", p.dur_min_ms, p.dur_max_ms, p.peak_min_g);
  Serial.printf("║     severe impulse:  >= %.3f g.s (delta-v %.2f m/s)\n", p.severe_impulse_gs, p.severe_impulse_gs * 9.80665);
  Serial.printf("║     server grading:  %s   local fallback grading: impulse\n", p.severity_grading);
  if (strcmp(p.name, "production") != 0) {
    Serial.println(F("║   !!! SCALED DEMONSTRATION PROFILE - NOT THE OPERATING TAXONOMY !!!"));
    Serial.println(F("║   !!! Events are excluded from all performance statistics      !!!"));
  }
}

void printBanner() {
  Serial.println();
  Serial.println(F("╔════════════════════════════════════════════════════════╗"));
  Serial.println(F("║   ESP32 COMPLETE CRASH DETECTION SYSTEM               ║"));
  Serial.println(F("║           WITH 3-COLOR LED ALERT SYSTEM               ║"));
  Serial.println(F("║                                                        ║"));
  Serial.println(F("║   Components:                                          ║"));
  Serial.println(F("║   ✅ MPU6050 (Crash Detection)                        ║"));
  Serial.println(F("║   ✅ WiFi (NESSA 1863)                                ║"));
  Serial.println(F("║   ✅ ML API (Severity Classification)                 ║"));
  Serial.println(F("║   ✅ LEDs (GPIO 33=RED, 25=YELLOW, 26=GREEN)          ║"));
  Serial.println(F("║   ✅ Buzzer (GPIO 32)                                 ║"));
  Serial.println(F("║                                                        ║"));
  Serial.println(F("║   LED Status Display:                                  ║"));
  Serial.println(F("║   🟢 GREEN = NORMAL (Safe - No action)                ║"));
  Serial.println(F("║   🟡 YELLOW = MODERATE (Warning - 3 pulses)           ║"));
  Serial.println(F("║   🔴 RED = SEVERE (Emergency - 5 rapid flashes)       ║"));
  Serial.println(F("║                                                        ║"));
  Serial.println(F("║   Buzzer Pattern:                                      ║"));
  Serial.println(F("║   🟢 NORMAL: 1 beep (1000 Hz)                         ║"));
  Serial.println(F("║   🟡 MODERATE: 3 beeps (1500 Hz)                      ║"));
  Serial.println(F("║   🔴 SEVERE: 5 beeps (2000 Hz - high pitch)           ║"));
  printProfileBanner();
  Serial.println(F("╚════════════════════════════════════════════════════════╝"));
  Serial.println();
}
