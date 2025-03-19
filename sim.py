import csv
import time
import random

OUTPUT_FILE = "sensor_data.csv"

def generate_sensor_data():
    """Simulates accelerometer & gyroscope readings."""
    return [
        round(random.uniform(-2, 2), 2),   # AccX
        round(random.uniform(-15, 15), 2), # AccY
        round(random.uniform(10, 15), 2),  # AccZ
        round(random.uniform(-10, 10), 2), # GyrX
        round(random.uniform(-5, 5), 2),   # GyrY
        round(random.uniform(-2, 2), 2)    # GyrZ
    ]

try:
    print("📡 Simulating sensor data... (Press Ctrl+C to stop)")

    while True:
        data_batch = []  # Store readings

        print("🔄 Collecting 400 simulated readings (18 sec)...")
        for i in range(400):  
            data = generate_sensor_data()
            data.append("NO")  # Add "Processed" flag as "NO"
            data_batch.append(data)
            print(f"✅ {i+1}/400 Data Collected: {data}")

            time.sleep(20 / 400)  # Maintain 20 Hz frequency

        if len(data_batch) == 400:
            print(f"📝 Writing {len(data_batch)} readings to CSV...")

            with open(OUTPUT_FILE, "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "Processed"])  
                writer.writerows(data_batch)

            print("✅ CSV Updated: sensor_data.csv with latest 400 readings.")

        # Pause for 2 seconds before next batch
        print("⏸️ Pausing for 2 seconds before next batch...")
        time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Stopped simulated data collection.")
