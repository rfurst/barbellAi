#include <Arduino_LSM9DS1.h>
#include <MadgwickAHRS.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <ArduinoBLE.h>

// Model (replace with your trained model)
#include "rep_detection_model.h"

#define SEQUENCE_LENGTH 50    // 0.42s window at 119Hz
#define NUM_SENSORS 6         // ax, ay, az, gx, gy, gz
#define TENSOR_ARENA_SIZE 4096

Madgwick filter;
float velocity = 0.0;
uint32_t rep_count = 0;
char current_exercise[20] = "Unknown";
uint8_t imu_data[SEQUENCE_LENGTH * NUM_SENSORS];
uint16_t data_index = 0;
unsigned long last_inference = 0;
bool is_moving = false;

// TFLite setup
const tflite::Model* model = tflite::GetModel(g_rep_detection_model);
tflite::AllOpsResolver resolver;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

// BLE Services
BLEService barbellService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic velChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, sizeof(float));
BLECharacteristic repChar("19B10002-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, sizeof(uint32_t));
BLECharacteristic exerciseChar("19B10003-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 20);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) while(1);
  if (!BLE.begin()) while(1);
  
  BLE.setLocalName("BarbellAI");
  BLE.setAdvertisedService(barbellService);
  barbellService.addCharacteristic(velChar);
  barbellService.addCharacteristic(repChar);
  barbellService.addCharacteristic(exerciseChar);
  BLE.addService(barbellService);
  BLE.advertise();

  if (interpreter.AllocateTensors() != kTfLiteOk) while(1);
  filter.begin(IMU.accelerationSampleRate());
}

void loop() {
  static unsigned long last_micros = micros();
  BLE.poll();

  float ax, ay, az, gx, gy, gz;
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // Normalize and store data
    imu_data[data_index * 6 + 0] = (ax / 4.0 * 127) + 128;
    imu_data[data_index * 6 + 1] = (ay / 4.0 * 127) + 128;
    imu_data[data_index * 6 + 2] = (az / 4.0 * 127) + 128;
    imu_data[data_index * 6 + 3] = (gx / 2000.0 * 127) + 128;
    imu_data[data_index * 6 + 4] = (gy / 2000.0 * 127) + 128;
    imu_data[data_index * 6 + 5] = (gz / 2000.0 * 127) + 128;
    data_index = (data_index + 1) % SEQUENCE_LENGTH;

    // Sensor fusion
    filter.update(gx * DEG_TO_RAD, gy * DEG_TO_RAD, gz * DEG_TO_RAD, ax, ay, az);
    float q0, q1, q2, q3;
    filter.getQuaternion(&q0, &q1, &q2, &q3);
    float vertical_accel = 2*(q1*q3 - q0*q2)*ax + 2*(q2*q3 + q0*q1)*ay + (2*(q0*q0 + q3*q3) -1)*az - 9.81;
    
    velocity += vertical_accel * (micros() - last_micros) / 1e6;
    last_micros = micros();
  }

  if (millis() - last_inference > 100) {
    // Run inference
    for (int i = 0; i < SEQUENCE_LENGTH * 6; i++) {
      interpreter.input(0)->data.int8[i] = imu_data[i] - 128;
    }
    if (interpreter.Invoke() != kTfLiteOk) return;

    float rep_prob = interpreter.output(0)->data.f[0];
    float velocity_correction = interpreter.output(0)->data.f[1];
    int exercise_id = (int)(interpreter.output(0)->data.f[2] + 0.5);

    velocity += velocity_correction;

    if (rep_prob > 0.7 && !is_moving) {
      is_moving = true;
      velocity = 0.0;
    } else if (rep_prob < 0.3 && is_moving) {
      is_moving = false;
      rep_count++;
      Serial.print("Rep "); Serial.print(rep_count); Serial.print(": "); Serial.println(velocity);
      
      velChar.writeValue(velocity);
      repChar.writeValue(rep_count);
      
      switch(exercise_id) {
        case 0: strcpy(current_exercise, "Deadlift"); break;
        case 1: strcpy(current_exercise, "Bench Press"); break;
        case 2: strcpy(current_exercise, "Squat"); break;
        default: strcpy(current_exercise, "Unknown");
      }
      exerciseChar.writeValue(current_exercise);
    }
    last_inference = millis();
  }
}