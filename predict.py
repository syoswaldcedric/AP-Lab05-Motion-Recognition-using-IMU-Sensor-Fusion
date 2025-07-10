import tensorflow as tf
import numpy as np
import time
from sense_hat import SenseHat

sense = SenseHat()
sense.clear()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="motion_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ["move_none", "move_circle", "move_shake", "move_twist"]
COLORS = {
    "move_none": [0, 0, 0],
    "move_circle": [255, 0, 0],
    "move_shake": [0, 255, 0],
    "move_twist": [0, 0, 255]
}
SAMPLES = 50
FREQ_HZ = 50
DELAY = 1.0 / FREQ_HZ

def read_imu_sample():
    acc = sense.get_accelerometer_raw()
    gyro = sense.get_gyroscope_raw()
    time.sleep(DELAY)
    return [acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']]

try:
    while True:
        print("Collecting 1s sample...")
        samples = [read_imu_sample() for _ in range(50)]
        input_data = np.array(samples).flatten().astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        end_time = time.time()
        predicted_index = int(np.argmax(output))
        label = LABELS[predicted_index]

        print(f"Predicted: {label} (pred. time: {end_time - start_time:.8f}s)")
        sense.clear(COLORS[label])
except KeyboardInterrupt:
    sense.clear()
    print("Stopped.")