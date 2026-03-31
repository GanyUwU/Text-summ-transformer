import sys
import os
import onnx
import numpy as np
from onnx import helper, TensorProto

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from debug_suite.core.model_loader import ModelLoader
from debug_suite.analyzers.weights import WeightAnalyzer
from debug_suite.analyzers.attention import AttentionAnalyzer
from debug_suite.analyzers.activations import ActivationAnalyzer
from debug_suite.analyzers.hallucination import HallucinationAnalyzer

def test_analyzers():
    print("\n=== Phase 2 Verification: Analyzers ===")
    
    # 1. Weight Analyzer Test
    # Create a dummy model with some weights
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    
    # Create a weight tensor (initializer) with one dead row
    W_val = np.array([[1.0, 2.0], [0.000001, 0.0]], dtype=np.float32)
    W = helper.make_tensor(name='W', data_type=TensorProto.FLOAT, dims=[2, 2], vals=W_val.flatten())
    
    node = helper.make_node('MatMul', inputs=['X', 'W'], outputs=['Y'])
    graph = helper.make_graph([node], 'test', [X], [Y], initializer=[W])
    model_def = helper.make_model(graph, producer_name='test')
    
    # Mock Loader
    class MockLoader:
        def __init__(self, m): self.model = m
    
    w_analyzer = WeightAnalyzer(MockLoader(model_def))
    w_report = w_analyzer.analyze()
    print("[Weights] Report keys:", w_report.keys())
    # Should detect 1 dead neuron row
    dead_count = w_report['summary']['dead_neurons']
    print(f"[Weights] Dead Neurons Detected: {dead_count}")
    if dead_count >= 1: print("[Weights] PASS")
    else: print("[Weights] FAIL - Expected dead neurons")
    
    # 2. Attention Analyzer Test
    # [batch, heads, seq, seq]
    # Head 0: Uniform (entropy high)
    # Head 1: Collapsed (entropy low)
    attn_data = np.zeros((1, 2, 4, 4), dtype=np.float32)
    attn_data[0, 0, :, :] = 0.25 # Uniform
    attn_data[0, 1, :, 0] = 1.0  # Collapsed to first token
    
    attn_analyzer = AttentionAnalyzer()
    a_report = attn_analyzer.analyze({"attn_layer": attn_data})
    print(f"[Attention] Head issues: {len(a_report.get('head_issues', []))}")
    if a_report['head_issues']: print("[Attention] PASS (Found issues)")
    else: print("[Attention] FAIL (No issues found)")

    # 3. Activation Analyzer Test
    # Create sparse activation (Relu output)
    act_data = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32) # 80% sparse
    act_analyzer = ActivationAnalyzer()
    act_report = act_analyzer.analyze({"relu_out": act_data})
    print(f"[Activation] Sparsity: {act_report['layer_stats']['relu_out']['sparsity']}")
    if act_report['layer_stats']['relu_out']['sparsity'] >= 0.8: print("[Activation] PASS")
    else: print("[Activation] FAIL")

    # 4. Hallucination Analyzer Test
    # [batch, seq, vocab]
    hall_data = np.zeros((1, 5, 10), dtype=np.float32)
    # Uniform logits -> High entropy -> High Risk
    hall_analyzer = HallucinationAnalyzer()
    h_report = hall_analyzer.analyze(hall_data)
    print(f"[Hallucination] Risk Score: {h_report['risk_score']:.1f}")
    if h_report['risk_score'] > 20: print("[Hallucination] PASS (High risk detected)")
    else: print("[Hallucination] FAIL (Risk too low)")

if __name__ == "__main__":
    test_analyzers()
