python -c "from ai_edge_litert.interpreter import Interpreter; i=Interpreter(model_path='/home/mvrop-avi/model_full_int8.tflite'); i.allocate_tensors(); print(i.get_input_details())"
