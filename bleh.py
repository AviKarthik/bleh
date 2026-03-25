from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import cv2
import time

MODEL_PATH = "/home/mvrop-avi/model.tflite"   # change if your filename is different
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

if len(input_shape) == 4:
    _, H, W, C = input_shape
    use_batch_dim = True
elif len(input_shape) == 3:
    H, W, C = input_shape
    use_batch_dim = False
else:
    raise ValueError(f"Unexpected input shape: {input_shape}")

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

def run_once():
    frame = picam2.capture_array()

    img = cv2.resize(frame, (W, H))

    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(input_dtype)

    if use_batch_dim:
        img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    if len(output.shape) > 1:
        output = output[0]

    pred_idx = int(np.argmax(output))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"
    confidence = float(output[pred_idx])

    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
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
