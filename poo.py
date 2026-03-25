python - << 'PY'
from ai_edge_litert.interpreter import Interpreter
i = Interpreter(model_path="/home/mvrop-avi/model.tflite")
i.allocate_tensors()
print(i.get_input_details())
PY
