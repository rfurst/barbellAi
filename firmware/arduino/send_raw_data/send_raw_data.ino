/********************************************************
  Arduino Code: Stream Raw IMU Data in Batches via BLE
********************************************************/
#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // For IMU on Nano 33 BLE Sense
#include <cstring>

// 1) Define your BLE Service/Characteristic UUIDs
#define SERVICE_UUID        "12340000-0000-0000-0000-1234567890AB"
#define CHARACTERISTIC_UUID "12340001-0000-0000-0000-1234567890AB"

// Use built-in LED for status indication
#define LED_PIN LED_BUILTIN

// IMU sampling rate (approx). We'll try ~100 Hz in the loop.
const int SAMPLE_DELAY_US = 10000; // 10 ms = 100 Hz

// How often we send a BLE packet (ms)
const unsigned long SEND_INTERVAL_MS = 10; // send every 10 ms

// Ring buffer settings
const int RING_BUFFER_SIZE = 128; // adjust as needed

struct ImuSample {
  uint32_t tMillis;  // 4 bytes
  float ax;          // 4 bytes
  float ay;          // 4 bytes
  float az;          // 4 bytes
  float gx;          // 4 bytes
  float gy;          // 4 bytes
  float gz;          // 4 bytes
};

// The ring buffer
ImuSample ringBuffer[RING_BUFFER_SIZE];
volatile int headIndex = 0; // where we write new samples
volatile int tailIndex = 0; // where we read for sending

// BLE objects
BLEService rawService(SERVICE_UUID);
BLECharacteristic rawDataChar(CHARACTERISTIC_UUID,
                             BLERead | BLENotify,
                             512); // max payload size (will do binary packing)

unsigned long lastSendMs = 0;

// Status tracking
bool bleInitialized = false;
bool imuInitialized = false;

void setup() {
  // Set up LED for status indication
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // Turn on LED at startup
  
  // Initialize Serial (but don't wait for it)
  Serial.begin(115200);
  
  // Add a longer startup delay to ensure stable power
  delay(2000);
  
  // Initialize BLE with multiple retries
  int retries = 0;
  while (!BLE.begin() && retries < 10) {
    delay(500);
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));  // Toggle LED during retry
    retries++;
  }
  
  if (retries < 10) {
    bleInitialized = true;
  }
  
  // Configure BLE regardless (attempt to recover)
  BLE.setLocalName("Nano33_RawIMU");
  BLE.setAdvertisedService(rawService);

  // Add characteristic to service
  rawService.addCharacteristic(rawDataChar);
  
  // Add service
  BLE.addService(rawService);

  // Start advertising
  BLE.advertise();
  
  // Initialize IMU with retries
  retries = 0;
  while (!IMU.begin() && retries < 10) {
    delay(500);
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));  // Toggle LED during retry
    retries++;
  }
  
  if (retries < 10) {
    imuInitialized = true;
  }
  
  // Status indicator
  if (bleInitialized && imuInitialized) {
    // Both initialized successfully - solid LED
    digitalWrite(LED_PIN, HIGH);
  } else if (bleInitialized) {
    // Only BLE initialized - fast blink
    for (int i = 0; i < 6; i++) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(100);
    }
  } else if (imuInitialized) {
    // Only IMU initialized - slow blink
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(200);
    }
  } else {
    // Nothing initialized - LED off
    digitalWrite(LED_PIN, LOW);
  }
  
  lastSendMs = millis();
}

/**
 * Write one sample into ring buffer (non-blocking).
 */
void storeSample(uint32_t tMs, float ax, float ay, float az,
                 float gx, float gy, float gz)
{
  int nextHead = (headIndex + 1) % RING_BUFFER_SIZE;
  // Check for overflow
  if (nextHead == tailIndex) {
    // Buffer full -> drop sample or move tail
    // For simplicity, let's overwrite oldest by moving tail
    tailIndex = (tailIndex + 1) % RING_BUFFER_SIZE;
  }
  ringBuffer[headIndex].tMillis = tMs;
  ringBuffer[headIndex].ax = ax;
  ringBuffer[headIndex].ay = ay;
  ringBuffer[headIndex].az = az;
  ringBuffer[headIndex].gx = gx;
  ringBuffer[headIndex].gy = gy;
  ringBuffer[headIndex].gz = gz;
  headIndex = nextHead;
}

/**
 * Send all buffered samples via a single BLE notification if space allows.
 * We'll pack them in binary form.
 */
void sendBufferBatch() {
  // We'll build a small binary payload in a local buffer.
  // Each sample is 7 fields * 4 bytes = 28 bytes.
  // We'll send as many as we can that fit in ~1 BLE packet (<=512 bytes).
  // In practice, your effective MTU might be ~185-247. You may need chunking.
  // This is a simplified example that tries to send up to 512 bytes at once.

  static uint8_t payload[512];
  int offset = 0;

  // Collect samples from tail to head
  while (tailIndex != headIndex) {
    if (offset + 28 > 512) {
      // no more space in this chunk
      break;
    }
    ImuSample &s = ringBuffer[tailIndex];
    // pack binary
    memcpy(payload + offset, &s.tMillis, 4); offset += 4;
    memcpy(payload + offset, &s.ax, 4);      offset += 4;
    memcpy(payload + offset, &s.ay, 4);      offset += 4;
    memcpy(payload + offset, &s.az, 4);      offset += 4;
    memcpy(payload + offset, &s.gx, 4);      offset += 4;
    memcpy(payload + offset, &s.gy, 4);      offset += 4;
    memcpy(payload + offset, &s.gz, 4);      offset += 4;

    // advance tail
    tailIndex = (tailIndex + 1) % RING_BUFFER_SIZE;
  }

  // Now if offset>0, we have data to send
  if (offset > 0) {
    rawDataChar.writeValue(payload, offset);
  }
}

// Attempt to recover BLE if it's not working
void attemptBLERecovery() {
  static unsigned long lastRecoveryAttempt = 0;
  unsigned long now = millis();
  
  // Only try recovery every 30 seconds
  if (now - lastRecoveryAttempt < 30000) {
    return;
  }
  
  lastRecoveryAttempt = now;
  
  // Toggle LED to indicate recovery attempt
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  
  // Try to re-initialize BLE
  BLE.end();
  delay(500);
  
  if (BLE.begin()) {
    bleInitialized = true;
    BLE.setLocalName("Nano33_RawIMU");
    BLE.setAdvertisedService(rawService);
    BLE.addService(rawService);
    BLE.advertise();
    digitalWrite(LED_PIN, HIGH);  // Success indicator
  } else {
    digitalWrite(LED_PIN, LOW);   // Failure indicator
  }
}

void loop() {
  // Check if BLE is still working, attempt recovery if needed
  if (!bleInitialized) {
    attemptBLERecovery();
  } else {
    // Service BLE connections
    BLEDevice central = BLE.central();
    
    // If central connected, blink LED
    static unsigned long lastBlink = 0;
    if (central && central.connected()) {
      if (millis() - lastBlink > 1000) {
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        lastBlink = millis();
      }
    }
  }

  // 1) Sample the IMU at ~100 Hz (block or time check).
  static unsigned long lastSampleMicros = 0;
  unsigned long nowMicros = micros();
  if (nowMicros - lastSampleMicros >= SAMPLE_DELAY_US && imuInitialized) {
    lastSampleMicros = nowMicros;

    // read IMU
    float ax, ay, az;
    float gx, gy, gz; // in deg/s by default
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      // store in buffer
      storeSample(millis(), ax, ay, az, gx, gy, gz);
    }
  }

  // 2) Every 10 ms, send all queued samples in one notification if BLE is working
  unsigned long nowMs = millis();
  if (nowMs - lastSendMs >= SEND_INTERVAL_MS && bleInitialized) {
    lastSendMs = nowMs;
    sendBufferBatch();
  }
}