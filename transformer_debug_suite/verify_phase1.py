import sys
import os
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from debug_suite.core.model_loader import ModelLoader
from debug_suite.core.graph_instrumenter import GraphInstrumenter
from debug_suite.core.inference_engine import InferenceEngine

def create_dummy_model(path: str):
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 5])
    
    node_relu = helper.make_node('Relu', inputs=['X'], outputs=['relu_out'], name='relu_node')
    node_add = helper.make_node('Add', inputs=['relu_out', 'Y'], outputs=['Z'], name='add_node')
    
    graph_def = helper.make_graph([node_relu, node_add], 'test-model', [X, Y], [Z])
    
    # Use older opset/IR for compatibility
    opset = helper.make_opsetid("", 11)
    # IR version 6 is widely supported (ONNX 1.5+)
    model_def = helper.make_model(graph_def, producer_name='debug_suite_test', 
                                  opset_imports=[opset], ir_version=6)
    
    onnx.save(model_def, path)

def test_phase_1():
    print("=== Phase 1 Verification (Conservative) ===")
    model_path = "dummy_model.onnx"
    instr_path = "instrumented_test.onnx"
    
    try:
        create_dummy_model(model_path)
        loader = ModelLoader(model_path)
        model = loader.load()
        print("Loader: OK")
        
        instrumenter = GraphInstrumenter(model)
        model, map_ = instrumenter.instrument(target_types={'Relu'})
        print(f"Instrumented Map: {map_}")
        
        onnx.save(model, instr_path)
        print("Instrumenter: OK")
        
        engine = InferenceEngine(instr_path)
        print("InferenceEngine: OK (loaded model)")
        
        x_val = np.array([[-1.0, 2.0, -3.0, 4.0, 5.0]], dtype=np.float32)
        y_val = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        
        res = engine.run({'X': x_val, 'Y': y_val})
        
        if 'relu_out' in res:
            print(f"Captured Internal Tensor (Relu): {res['relu_out']}")
            print("SUCCESS: Phase 1 Operational")
        else:
            print(f"FAIL: Available outputs: {list(res.keys())}")
            exit(1)
            
    except Exception as e:
        print(f"FAIL with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        for p in [model_path, instr_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

if __name__ == "__main__":
    test_phase_1()
