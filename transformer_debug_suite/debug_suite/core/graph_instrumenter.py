import onnx
from onnx import helper, ModelProto, NodeProto, ValueInfoProto
from typing import List, Dict, Set

class GraphInstrumenter:
    """
    Modifies an ONNX model to expose internal intermediate tensors as outputs.
    This enables runtime inspection of Attention maps, Activations, and Norms.
    """
    
    def __init__(self, model: ModelProto):
        self.model = model
        self.graph = model.graph
        
        # Tensor names that are already outputs
        self.existing_outputs = {out.name for out in self.graph.output}
        
    def instrument(self, target_types: Set[str] = None) -> ModelProto:
        """
        Instruments the model to expose outputs of nodes matching target types.
        
        Args:
            target_types: Set of OP types to instrument. 
                          Defaults to {'Softmax', 'LayerNormalization', 'Relu', 'Gelu', 'Add'}.
                          
        Returns:
            ModelProto: The modified model with new outputs.
        """
        if target_types is None:
            # Default to extracting Attention (Softmax), Norms, and Activations
            target_types = {'Softmax', 'LayerNormalization', 'Relu', 'Gelu', 'Add', 'Gemm', 'MatMul'}
            
        new_outputs: List[ValueInfoProto] = []
        exposed_tensors: Dict[str, str] = {} # output_name -> node_type
        
        print(f"Instrumenting graph for types: {target_types}")
        
        # Optional: Run shape inference to get better type info
        try:
            inferred_model = onnx.shape_inference.infer_shapes(self.model)
            value_info = {vi.name: vi for vi in inferred_model.graph.value_info}
            # Also include existing outputs and inputs in the lookup
            for i in inferred_model.graph.input: value_info[i.name] = i
            for o in inferred_model.graph.output: value_info[o.name] = o
        except Exception as e:
            print(f"Warning: Shape inference failed: {e}")
            value_info = {}

        for node in self.graph.node:
            if node.op_type in target_types:
                for output_name in node.output:
                    if output_name not in self.existing_outputs:
                        # Determine the element type (default to FLOAT if unknown)
                        elem_type = onnx.TensorProto.FLOAT
                        if output_name in value_info:
                            elem_type = value_info[output_name].type.tensor_type.elem_type
                        
                        output_info = onnx.helper.make_tensor_value_info(
                            name=output_name,
                            elem_type=elem_type,
                            shape=None 
                        )
                        new_outputs.append(output_info)
                        exposed_tensors[output_name] = node.op_type
        
        # Add new outputs to graph
        self.graph.output.extend(new_outputs)
        
        print(f"Exposed {len(new_outputs)} internal tensors.")
        return self.model, exposed_tensors

    def save_instrumented(self, path: str):
        """Saves the modified model to disk."""
        onnx.save(self.model, path)
        print(f"Saved instrumented model to {path}")
