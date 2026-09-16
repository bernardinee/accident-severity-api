// Crash-signature gate, identical to the server's profiles.crash_signature() applied
// after app.py's 20 Hz sosfiltfilt. Plain C++ (no Arduino headers) so the host test
// tests/test_firmware_gate.py can compile it and compare against the Python gate.
#pragma once
#include <cmath>
#include "threshold_profiles.h"

static const int GATE_WINDOW = 500;

struct GateResult {
  float peak_g;
  float excursion_ms;
  float impulse_gs;
  bool signature_match;
  int severity_class;   // 0 Normal, 1 Moderate, 2 Severe (impulse grading)
};

// scipy.signal.sosfilt, transposed direct form II, with initial state zi * x0.
inline void gate_sosfilt(double* x, int n, double x0) {
  double z[GATE_SOS_SECTIONS][2];
  for (int s = 0; s < GATE_SOS_SECTIONS; s++) {
    z[s][0] = GATE_ZI[s][0] * x0;
    z[s][1] = GATE_ZI[s][1] * x0;
  }
  for (int i = 0; i < n; i++) {
    double cur = x[i];
    for (int s = 0; s < GATE_SOS_SECTIONS; s++) {
      const double* c = GATE_SOS[s];
      double y = c[0] * cur + z[s][0];
      z[s][0] = c[1] * cur - c[4] * y + z[s][1];
      z[s][1] = c[2] * cur - c[5] * y;
      cur = y;
    }
    x[i] = cur;
  }
}

inline void gate_reverse(double* x, int n) {
  for (int i = 0, j = n - 1; i < j; i++, j--) { double t = x[i]; x[i] = x[j]; x[j] = t; }
}

// scipy.signal.sosfiltfilt(sos, in, padtype='odd', padlen=GATE_PADLEN) into out[n].
inline void gate_filtfilt(const float* in, double* out, int n) {
  static double ext[GATE_WINDOW + 2 * GATE_PADLEN];
  const int P = GATE_PADLEN, N = n + 2 * P;
  for (int i = 0; i < P; i++) ext[i] = 2.0 * in[0] - in[P - i];                 // 2*x[0] - x[P..1]
  for (int i = 0; i < n; i++) ext[P + i] = in[i];
  for (int i = 0; i < P; i++) ext[P + n + i] = 2.0 * in[n - 1] - in[n - 2 - i];  // 2*x[-1] - x[-2..]
  gate_sosfilt(ext, N, ext[0]);
  gate_reverse(ext, N);
  gate_sosfilt(ext, N, ext[0]);
  gate_reverse(ext, N);
  for (int i = 0; i < n; i++) out[i] = ext[P + i];
}

// ax/ay/az: GATE_WINDOW samples in g, oldest first.
inline GateResult evaluateSignature(const float* ax, const float* ay, const float* az,
                                    const ThresholdProfile& p) {
  static double fx[GATE_WINDOW], fy[GATE_WINDOW], fz[GATE_WINDOW];
  gate_filtfilt(ax, fx, GATE_WINDOW);
  gate_filtfilt(ay, fy, GATE_WINDOW);
  gate_filtfilt(az, fz, GATE_WINDOW);
  double peak = 0, impulse = 0;
  int run = 0, longest = 0;
  for (int i = 0; i < GATE_WINDOW; i++) {
    double m = std::sqrt(fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i]);
    if (m > peak) peak = m;
    if (m >= p.peak_min_g) {
      run++;
      impulse += m - 1.0;
      if (run > longest) longest = run;
    } else {
      run = 0;
    }
  }
  GateResult r;
  r.peak_g = (float)peak;
  r.excursion_ms = longest * 1000.0f / GATE_FS_HZ;
  r.impulse_gs = (float)(impulse / GATE_FS_HZ);
  r.signature_match = peak >= p.peak_min_g && peak < p.peak_max_g &&
                      r.excursion_ms >= p.dur_min_ms && r.excursion_ms <= p.dur_max_ms;
  r.severity_class = !r.signature_match ? 0 : (r.impulse_gs >= p.severe_impulse_gs ? 2 : 1);
  return r;
}
