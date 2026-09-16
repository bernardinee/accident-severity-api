// Host build of the firmware gate for tests/test_firmware_gate.py.
//   harness --profile                      print the compiled-in profile
//   harness [pmin pmax dmin dmax severe]   read windows (1500 floats: ax*500 ay*500 az*500)
//                                          from stdin, print "peak excursion impulse sig class"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../firmware/ESP32_Complete_3LED_System/signature_gate.h"

int main(int argc, char** argv) {
  ThresholdProfile p = ACTIVE_THRESHOLD_PROFILE;
  if (argc == 2 && strcmp(argv[1], "--profile") == 0) {
    printf("%s|%s|%s|%.6g|%.6g|%.6g|%.6g|%.6g|%s\n", p.name, p.requested, p.note ? p.note : "",
           p.peak_min_g, p.peak_max_g, p.dur_min_ms, p.dur_max_ms, p.severe_impulse_gs, p.severity_grading);
    return 0;
  }
  if (argc == 6) {
    p.peak_min_g = atof(argv[1]); p.peak_max_g = atof(argv[2]);
    p.dur_min_ms = atof(argv[3]); p.dur_max_ms = atof(argv[4]); p.severe_impulse_gs = atof(argv[5]);
  }
  static float ax[GATE_WINDOW], ay[GATE_WINDOW], az[GATE_WINDOW];
  float* ch[3] = {ax, ay, az};
  while (true) {
    for (int c = 0; c < 3; c++)
      for (int i = 0; i < GATE_WINDOW; i++)
        if (scanf("%f", &ch[c][i]) != 1) return 0;
    GateResult r = evaluateSignature(ax, ay, az, p);
    printf("%.9g %.9g %.9g %d %d\n", r.peak_g, r.excursion_ms, r.impulse_gs, r.signature_match ? 1 : 0,
           r.severity_class);
  }
}
