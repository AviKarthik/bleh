from picamera2 import Picamera2
import cv2
import numpy as np
from flask import Flask, Response

from ai_edge_litert.interpreter import Interpreter

# ===== MODEL =====
MODEL_PATH = "model_full_int8.tflite"
LABELS_PATH = "labels.txt"

labels = [l.strip() for l in open(LABELS_PATH)]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_index = input_details[0]['index']
out_index = output_details[0]['index']

# ===== CAMERA =====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (224, 224)}))
picam2.start()

# ===== FLASK =====
app = Flask(__name__)

def generate():
    while True:
        frame = picam2.capture_array()

        img = frame.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # quantize
        x_q = np.clip(img, -128, 127).astype(np.int8)

        interpreter.set_tensor(in_index, x_q)
        interpreter.invoke()
        y = interpreter.get_tensor(out_index)

        pred_idx = int(np.argmax(y))
        label = labels[pred_idx]

        # draw text
        cv2.putText(frame, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video():
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)
