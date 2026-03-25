from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import cv2
import time

MODEL_PATH = "/home/mvrop-avi/model_full_int8.tflite"
LABELS_PATH = "/home/mvrop-avi/labels.txt"

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f if line.strip()]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

print("Input shape:", input_shape)
print("Input dtype:", input_dtype)

_, H, W, C = input_shape

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

def run_once():
    frame = picam2.capture_array()

    img = cv2.resize(frame, (W, H))

    # convert to int8 format expected by model
    img = img.astype(np.int16) - 128
    img = img.astype(np.int8)

    # add batch dimension (REQUIRED for your model)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(output))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"
    confidence = int(output[pred_idx])

    print(f"Prediction: {pred_label}")
    print(f"Raw output value: {confidence}")
    print()

print("Ready.")
print("Press ENTER to capture and classify.")
print("Type q then ENTER to quit.")

while True:
    cmd = input("> ")
    if cmd.strip().lower() == "q":
        break
    run_once()

picam2.stop()
