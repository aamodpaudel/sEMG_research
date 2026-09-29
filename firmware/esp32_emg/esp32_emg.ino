// Afono $50 rig firmware (Phase 4, UNTESTED on hardware yet).
// ESP32 dev board + up to 4 BioAmp EXG Pill boards on ADC1 pins.
// Streams one CSV line per sample at 1000 Hz over USB serial, 921600 baud:
//   c1,c2,c3,c4\n
// Afono Lab reads this with Web Serial ("Connect sensor").
// SAFETY: power the rig from a battery or an unplugged laptop only. Never from mains while on skin.

const int PINS[] = {36, 39, 34, 35};   // ADC1 channels (ADC2 is unusable with WiFi)
const int N = sizeof(PINS) / sizeof(PINS[0]);
const uint32_t PERIOD_US = 1000;       // 1000 Hz

uint32_t next_t;

void setup() {
  Serial.begin(921600);
  analogReadResolution(12);            // 0..4095
  for (int i = 0; i < N; i++) analogSetPinAttenuation(PINS[i], ADC_11db);
  next_t = micros();
}

void loop() {
  if ((int32_t)(micros() - next_t) < 0) return;
  next_t += PERIOD_US;
  char line[40];
  int len = 0;
  for (int i = 0; i < N; i++) {
    len += snprintf(line + len, sizeof(line) - len, i ? ",%d" : "%d", analogRead(PINS[i]));
  }
  line[len++] = '\n';
  Serial.write((const uint8_t*)line, len);
}
