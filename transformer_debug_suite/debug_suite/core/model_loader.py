import onnx
import os
from typing import Optional, Tuple

class ModelLoader:
    """
    Handles loading and validation of ONNX models.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load(self) -> onnx.ModelProto:
        """
        Loads the ONNX model from disk.
        
        Returns:
            onnx.ModelProto: The loaded ONNX model.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
            onnx.checker.ValidationError: If model is invalid.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
        print(f"Loading model from {self.model_path}...")
        self.model = onnx.load(self.model_path)
        
        # Validate model structure
        print("Validating model structure...")
        onnx.checker.check_model(self.model)
        print("Model validation passed.")
        
        return self.model

    def get_metadata(self) -> dict:
        """
        Returns basic metadata about the loaded model.
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        return {
            "ir_version": self.model.ir_version,
            "producer_name": self.model.producer_name,
            "producer_version": self.model.producer_version,
            "graph_name": self.model.graph.name,
            "inputs": [node.name for node in self.model.graph.input],
            "outputs": [node.name for node in self.model.graph.output],
        }
