import onnxruntime as ort
import numpy as np
from typing import Dict, List, Any, Union

class InferenceEngine:
    """
    Wraps ONNX Runtime to execute instrumented models.
    """
    
    def __init__(self, model_path: Union[str, bytes]):
        """
        Args:
            model_path: Path to ONNX file OR bytes of ONNX model.
        """
        # ONNX Runtime can load from bytes directly, which is useful 
        # so we don't have to save the instrumented model to disk every time.
        self.session = ort.InferenceSession(model_path)
        
        self.inputs = [x.name for x in self.session.get_inputs()]
        self.outputs = [x.name for x in self.session.get_outputs()]
        
    def run(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Runs inference and returns all outputs.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays.
            
        Returns:
            Dict mapping output names (including exposed internals) to numpy arrays.
        """
        # Validate inputs
        for inp_name in self.inputs:
            if inp_name not in input_data:
                raise ValueError(f"Missing input: {inp_name}")
                
        # Run
        raw_results = self.session.run(self.outputs, input_data)
        
        # Map back to names
        results = {name: val for name, val in zip(self.outputs, raw_results)}
        return results
