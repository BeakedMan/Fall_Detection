#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// Calibration offsets (updated dynamically)
float ax_bias = 0, ay_bias = 0, az_bias = 0;
float gx_bias = 0, gy_bias = 0, gz_bias = 0;
const int calibrationSamples = 500; // Number of samples for calibration

void calibrateSensor() {
    Serial.println("🔄 Starting Calibration... Keep the sensor STILL on a flat surface.");

    for (int i = 0; i < calibrationSamples; i++) {
        sensors_event_t accel, gyro, temp;
        mpu.getEvent(&accel, &gyro, &temp);

        ax_bias += accel.acceleration.x;
        ay_bias += accel.acceleration.y;
        az_bias += (accel.acceleration.z - 9.81);  // Adjust for gravity
        gx_bias += gyro.gyro.x;
        gy_bias += gyro.gyro.y;
        gz_bias += gyro.gyro.z;

        delay(5);  // Short delay to get accurate readings
    }

    ax_bias /= calibrationSamples;
    ay_bias /= calibrationSamples;
    az_bias /= calibrationSamples;
    gx_bias /= calibrationSamples;
    gy_bias /= calibrationSamples;
    gz_bias /= calibrationSamples;

    Serial.println("✅ Calibration Complete! Computed Offsets:");
    Serial.print("Accel Bias: "); Serial.print(ax_bias); Serial.print(", ");
    Serial.print(ay_bias); Serial.print(", "); Serial.println(az_bias);
    Serial.print("Gyro Bias: "); Serial.print(gx_bias); Serial.print(", ");
    Serial.print(gy_bias); Serial.print(", "); Serial.println(gz_bias);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!mpu.begin()) {
        Serial.println("❌ Failed to initialize MPU6050!");
        while (1);
    }

    Serial.println("✅ MPU6050 initialized successfully!");

    // Set ranges according to dataset values
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // Run self-calibration
    calibrateSensor();
}

void loop() {
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

    // Apply dynamically computed Bias Correction
    float ax = accel.acceleration.x - ax_bias;
    float ay = accel.acceleration.y - ay_bias;
    float az = accel.acceleration.z - az_bias;
    float gx = gyro.gyro.x - gx_bias;
    float gy = gyro.gyro.y - gy_bias;
    float gz = gyro.gyro.z - gz_bias;

    // Print values in CSV format for Python script
    Serial.print(ax); Serial.print(",");
    Serial.print(ay); Serial.print(",");
    Serial.print(az); Serial.print(",");
    Serial.print(gx); Serial.print(",");
    Serial.print(gy); Serial.print(",");
    Serial.println(gz);

    delay(50); // Maintain 20 Hz frequency (50ms per reading)
}