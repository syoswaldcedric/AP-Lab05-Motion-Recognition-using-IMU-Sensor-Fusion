import os
import time
import numpy as np
from sense_hat import SenseHat

sense = SenseHat()
LABEL = "move_circle"  # change to "move_shake", "move_twist" or "move_none" for other classes
SAMPLES = 50  # 50 for 1s
FREQ_HZ = 50
DELAY = 1.0 / FREQ_HZ

save_dir = f"./motion_data/{LABEL}"
os.makedirs(save_dir, exist_ok=True)

print(f"Recording samples for label: {LABEL}")
try:
    while True:
        input("Press Enter to record 1 second...")
        data = []
        for _ in range(SAMPLES):
            acc = sense.get_accelerometer_raw()
            gyro = sense.get_gyroscope_raw()
            sample = [
                acc['x'], acc['y'], acc['z'],
                gyro['x'], gyro['y'], gyro['z']
            ]
            data.append(sample)
            time.sleep(DELAY)
        timestamp = int(time.time())
        np.save(f"{save_dir}/{LABEL}_{timestamp}.npy", np.array(data))
        print(f"Saved {LABEL}_{timestamp}.npy")
except KeyboardInterrupt:
    print("Recording stopped.")