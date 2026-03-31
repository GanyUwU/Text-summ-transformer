import os
import sys
from onnx import helper, TensorProto
import onnx

# Create dummy model for end-to-end test
def create_test_model(path):
    # X, Y inputs
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 10])
    
    # Relu (Activation)
    node_relu = helper.make_node('Relu', ['X'], ['relu_out'], name='relu_n')
    # Softmax (Attention-like)
    node_soft = helper.make_node('Softmax', ['relu_out'], ['soft_out'], name='soft_n')
    # Add (Output)
    node_add = helper.make_node('Add', ['soft_out', 'Y'], ['Z'], name='add_n')
    
    graph = helper.make_graph([node_relu, node_soft, node_add], 'test', [X, Y], [Z])
    opset = helper.make_opsetid("", 11)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=6)
    onnx.save(model, path)

if __name__ == "__main__":
    model_path = "e2e_model.onnx"
    create_test_model(model_path)
    
    print("Running End-to-End Test...")
    
    # Run CLI via system command to test entry point logic
    cmd = f"python transformer_debug_suite/debug_suite/main.py {model_path} --output e2e_report.html"
    ret = os.system(cmd)
    
    if ret == 0 and os.path.exists("e2e_report.html"):
        print("\nSUCCESS: End-to-End Test Passed!")
        print("Report generated: e2e_report.html")
    else:
        print(f"\nFAIL: Return code {ret}")
        exit(1)
        
    # Cleanup
    # if os.path.exists(model_path): os.remove(model_path)
