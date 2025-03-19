import serial
import csv
import time

SERIAL_PORT = "COM5"  # Update to your correct port (Check Arduino IDE)
BAUD_RATE = 115200
OUTPUT_FILE = "sensor_data.csv"

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("📡 Reading self-calibrated sensor data from Arduino... (Press Ctrl+C to stop)")

    while True:
        data_batch = []  # Store valid readings

        print("🔄 Collecting 400 readings from Arduino (20 sec)...")

        while len(data_batch) < 400:  # Ensure exactly 400 valid readings
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if line:
                data = line.split(",")

                if len(data) == 6:  # Valid format (6 values per row)
                    try:
                        # Convert to float (ensure valid numerical data)
                        data = [float(x) for x in data]
                        data.append("NO")  # Add "Processed" flag
                        data_batch.append(data)
                        print(f"✅ {len(data_batch)}/400 Data Collected: {data}")
                    except ValueError:
                        print(f"⚠ Data conversion error: {line}. Skipping.")

                else:
                    print(f"⚠ Invalid data format: {line}. Skipping.")

            time.sleep(1 / 20)  # Maintain 20 Hz frequency (50ms per reading)

        # Ensure we have exactly 400 valid readings before writing
        print(f"📝 Writing {len(data_batch)} valid readings to CSV...")

        with open(OUTPUT_FILE, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "Processed"])  
            writer.writerows(data_batch)

        print("✅ CSV Updated: sensor_data.csv with latest 400 readings.")

        # Pause for 2 seconds before next batch
        print("⏸ Pausing for 2 seconds before next batch...")
        time.sleep(2)

except serial.SerialException as e:
    print(f"❌ Serial Error: {e}")

except KeyboardInterrupt:
    print("\n🛑 Stopped data collection.")
    ser.close()