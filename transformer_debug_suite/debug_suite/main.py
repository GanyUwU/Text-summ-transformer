import argparse
import sys
import os
import numpy as np
import onnx

# Add parent path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_debug_suite.debug_suite.core.model_loader import ModelLoader
from transformer_debug_suite.debug_suite.core.graph_instrumenter import GraphInstrumenter
from transformer_debug_suite.debug_suite.core.inference_engine import InferenceEngine

from transformer_debug_suite.debug_suite.analyzers.weights import WeightAnalyzer
from transformer_debug_suite.debug_suite.analyzers.attention import AttentionAnalyzer
from transformer_debug_suite.debug_suite.analyzers.activations import ActivationAnalyzer
from transformer_debug_suite.debug_suite.analyzers.hallucination import HallucinationAnalyzer

from transformer_debug_suite.debug_suite.visualization.html_generator import HTMLGenerator

class DebugSuiteCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Universal Transformer Debug Suite")
        self.parser.add_argument("model_path", help="Path to ONNX model")
        self.parser.add_argument("--batch-size", type=int, default=1)
        self.parser.add_argument("--seq-len", type=int, default=10) # Small default for testing
        self.parser.add_argument("--output", default="debug_report.html", help="Output path for HTML report")
        self.parser.add_argument("--input-text", default="", help="Real text for semantic analysis")

    def run(self):
        args = self.parser.parse_args()
        print(f"=== Starting Debug Suite for {args.model_path} ===")
        
        # 1. Load
        try:
            loader = ModelLoader(args.model_path)
            model = loader.load()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
            
        # 2. Static Analysis (Weights)
        print("Running Static Analysis...")
        w_analyzer = WeightAnalyzer(loader)
        w_results = w_analyzer.analyze()
        
        # 3. Instrument
        print("Instrumenting Graph...")
        instrumenter = GraphInstrumenter(model)
        target_types = {'Softmax', 'Relu', 'Gelu', 'Add', 'MatMul'}
        model, exposed_map = instrumenter.instrument(target_types)
        
        temp_path = "temp_instr.onnx"
        try:
            onnx.save(model, temp_path)
            
            # 4. Inference
            print("Running Dynamic Analysis...")
            engine = InferenceEngine(temp_path)
            
            # Generate inputs intelligently based on metadata
            inputs = {}
            for inp in engine.session.get_inputs():
                # Use seq-len for dynamic axes (usually index 1)
                shape = [d if isinstance(d, int) else args.seq_len for d in inp.shape]
                # Fallback for batch size if dynamic
                if not isinstance(shape[0], int): shape[0] = args.batch_size
                
                if 'int' in inp.type:
                    dtype = np.int64 if '64' in inp.type else np.int32
                    inputs[inp.name] = np.random.randint(0, 500, size=shape).astype(dtype)
                elif 'bool' in inp.type:
                    inputs[inp.name] = (np.random.rand(*shape) > 0.5)
                else:
                    inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

            outputs = engine.run(inputs)
            print(f"Captured {len(outputs)} tensors.")
            
            # 5. Analyze Outputs
            attn_tensors = {k:v for k,v in outputs.items() if exposed_map.get(k) == 'Softmax'}
            act_tensors = {k:v for k,v in outputs.items() if exposed_map.get(k) in ['Relu','Gelu','Add']}
            # Assume last output is logits for simplicity
            logits = list(outputs.values())[-1] 
            
            # Use real text if provided for SBERT
            input_text = getattr(args, 'input_text', "The quick brown fox jumps over the lazy dog.")

            attn_results = AttentionAnalyzer().analyze(attn_tensors)
            act_results = ActivationAnalyzer().analyze(act_tensors)
            
            from transformer_debug_suite.debug_suite.analyzers.semantic import SemanticAnalyzer
            semantic_results = SemanticAnalyzer().analyze(act_tensors, input_text)
            
            hall_results = {}
            if len(logits.shape) == 3:
                hall_results = HallucinationAnalyzer().analyze(logits)
            
            # 6. Report
            all_issues = []
            all_issues.extend(w_results.get("issues", []))
            all_issues.extend(attn_results.get("head_issues", []))
            all_issues.extend(act_results.get("issues", []))
            all_issues.extend(hall_results.get("issues", []))
            all_issues.extend(semantic_results.get("issues", []))
            
            results = {
                "weights": w_results,
                "attention": attn_results, 
                "activations": act_results,
                "hallucination": hall_results,
                "semantic": semantic_results,
                "all_issues": all_issues
            }
            
            HTMLGenerator().generate(results, args.output)
            print(f"Report saved to {args.output}")
            
        except Exception as e:
            print(f"Runtime Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

if __name__ == "__main__":
    DebugSuiteCLI().run()
