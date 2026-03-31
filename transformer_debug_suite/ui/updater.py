import os
import re

file_path = "c:/Users/prathamesh/Desktop/Text summ transformer/transformer_debug_suite/ui/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add imports
imports_to_add = """from debug_suite.analyzers.model_health import ModelHealthAnalyzer
from debug_suite.analyzers.weight_evolution import WeightEvolutionAnalyzer
from debug_suite.analyzers.gradient_flow import GradientFlowAnalyzer
import torch

try:
    from model import build_transformer
    import pretrain_config
    _HAS_TORCH_MODEL = True
except ImportError:
    _HAS_TORCH_MODEL = False
"""
text = text.replace("from debug_suite.analyzers.model_health import ModelHealthAnalyzer", imports_to_add)

# 2. XAI explanation additions
xai_additions = """
    "gradients": \"\"\"
## Gradient Flow Explained

**What we measure:**
- **Simulated Backward Pass**: We run synthetic data through the model to check backpropagation.
- **Vanishing Gradients**: Standard deviation < 1e-6. Signal dies before reaching early layers.
- **Exploding Gradients**: Standard deviation > 10.0. Training becomes unstable.

**How to fix:**
- **Vanishing**: Check for dying ReLUs or apply Xavier/Kaiming initialization. Remove excessive LayerNorms.
- **Exploding**: Apply gradient clipping (`clip_grad_norm_`) or reduce learning rate.
\"\"\",
    "evolution": \"\"\"
## Weight Evolution Explained

**What we measure:**
- **L2 Distance over Epochs**: How much the weights are actually moving during training.

**How to fix:**
- **Frozen Encoder**: If distance < 1e-4, the learning rate is too low or gradients are vanishing.
- **Erratic Movement**: If distance spikes wildly, lower learning rate or increase batch size.
\"\"\",
"""
text = text.replace('    "model_health": """', xai_additions + '    "model_health": """')

# 3. Modify analyze_model inputs/outputs
old_analyze_start = """def analyze_model(file, input_text):
    \"\"\"Main analysis function called when user uploads and clicks Analyze.\"\"\"
    global analysis_results, current_model_path
    
    if file is None:
        return " Please upload an ONNX model first.", None, None, None, None, None, None
    
    if not input_text:
        input_text = "The quick brown fox jumps over the lazy dog."
    
    try:
        # Save uploaded file
        current_model_path = file.name
        
        # 1. Load Model
        loader = ModelLoader(current_model_path)"""

new_analyze_start = """def analyze_model(file, input_text):
    \"\"\"Main analysis function called when user uploads and clicks Analyze.\"\"\"
    global analysis_results, current_model_path
    
    if file is None:
        return (" Please upload an ONNX or .pt model first.",) + (None,) * 12
    
    is_pt = file.name.endswith(".pt")
    is_onnx = file.name.endswith(".onnx")
    
    if not (is_pt or is_onnx):
         return (" Please upload a .onnx or .pt file.",) + (None,) * 12
         
    if not input_text:
        input_text = "The quick brown fox jumps over the lazy dog."
    
    try:
        current_model_path = file.name
        
        weight_evo_plot, grad_flow_plot = None, None
        
        if is_pt and _HAS_TORCH_MODEL:
            try:
                config = pretrain_config.get_finetune_config()
                model_pt = build_transformer(
                    src_vocab_size=32000, tgt_vocab_size=32000,
                    src_seq_len=config['src_seq_len'], tgt_seq_len=config['tgt_seq_len'],
                    d_model=config['d_model'], N=config['n_layers'], h=config['n_heads'],
                    dropout=config['dropout'], d_ff=config['d_ff'],
                    share_weights=config['share_weights'], use_copy=config['use_copy']
                )
                ckpt = torch.load(current_model_path, map_location='cpu')
                model_pt.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
                
                grad_results = GradientFlowAnalyzer(model_pt).analyze()
                analysis_results["gradients"] = grad_results
                
                ckpt_dir = os.path.dirname(current_model_path)
                evo_results = WeightEvolutionAnalyzer(ckpt_dir).analyze(max_checkpoints=10)
                analysis_results["evolution"] = evo_results
            except Exception as e:
                print(f"Deep PyTorch Analysis Failed: {e}")
                
        if is_onnx:
            # 1. Load Model
            loader = ModelLoader(current_model_path)"""
text = text.replace(old_analyze_start, new_analyze_start)

# Fast forward to outputs logic
old_outputs_logic = """        # Store results
        analysis_results = {
            "metadata": metadata,
            "weights": w_results,
            "attention": attn_results,
            "activations": act_results,
            "hallucination": hall_results,
            "semantic": semantic_results,
            "health": health_results,
            "raw_outputs": outputs,
            "exposed_map": exposed_map
        }
        
        # Cleanup
        os.remove(temp_path)
        
        # Generate outputs for each tab
        overview = generate_overview()
        weight_plot = generate_weight_plot()
        attn_plot = generate_attention_plot()
        act_plot = generate_activation_plot()
        hall_plot = generate_hallucination_plot()
        semantic_plot = generate_semantic_plot()
        health_heatmap = generate_health_heatmap()
        health_dashboard = generate_health_dashboard()
        health_signal = generate_health_signal()
        health_summary = generate_health_summary_md()
        
        return (overview, weight_plot, attn_plot, act_plot, hall_plot,
                semantic_plot, health_heatmap, health_dashboard,
                health_signal, health_summary, "Analysis complete!")
        
    except Exception as e:
        import traceback
        return (f" Error: {str(e)}\\n\\n{traceback.format_exc()}",
                None, None, None, None, None, None, None, None, None, None)"""

new_outputs_logic = """        # Store results
        if is_onnx:
            analysis_results.update({
                "metadata": metadata,
                "weights": w_results,
                "attention": attn_results,
                "activations": act_results,
                "hallucination": hall_results,
                "semantic": semantic_results,
                "health": health_results,
                "raw_outputs": outputs,
                "exposed_map": exposed_map
            })
            os.remove(temp_path)
        else:
            analysis_results.update({"metadata": {"inputs": [], "outputs": []}, "weights": {"summary": {}}})
        
        overview = generate_overview()
        weight_plot = generate_weight_plot() if is_onnx else None
        attn_plot = generate_attention_plot() if is_onnx else None
        act_plot = generate_activation_plot() if is_onnx else None
        hall_plot = generate_hallucination_plot() if is_onnx else None
        semantic_plot = generate_semantic_plot() if is_onnx else None
        health_heatmap = generate_health_heatmap() if is_onnx else None
        health_dashboard = generate_health_dashboard() if is_onnx else None
        health_signal = generate_health_signal() if is_onnx else None
        health_summary = generate_health_summary_md() if is_onnx else None
        
        evo_plot = generate_evolution_plot()
        grad_plot = generate_gradient_plot()
        
        return (overview, weight_plot, attn_plot, act_plot, hall_plot,
                semantic_plot, health_heatmap, health_dashboard,
                health_signal, health_summary, evo_plot, grad_plot, "Analysis complete!")
        
    except Exception as e:
        import traceback
        return (f" Error: {str(e)}\\n\\n{traceback.format_exc()}",) + (None,) * 12"""
text = text.replace(old_outputs_logic, new_outputs_logic)

# Indent ONNX specific code in analyze_model
old_onnx_block = """        # 2. Weight Analysis (Static)
        w_analyzer = WeightAnalyzer(loader)
        w_results = w_analyzer.analyze()
        
        # 3. Instrument Graph
        instrumenter = GraphInstrumenter(model)
        model_instr, exposed_map = instrumenter.instrument({'Softmax', 'Relu', 'Gelu', 'Add'})
        
        # Save instrumented model temporarily
        import onnx
        temp_path = tempfile.mktemp(suffix='.onnx')
        onnx.save(model_instr, temp_path)
        
        # 4. Run Inference
        engine = InferenceEngine(temp_path)
        
        # Generate dtype-correct dummy inputs based on model inputs
        inputs = {}
        for inp in engine.session.get_inputs():
            # Use 32 as default for dynamic axes to ensure valid attention entropy (log(seq_len) > 0)
            shape = [d if isinstance(d, int) else 32 for d in inp.shape]
            t = inp.type.lower()

            if "bool" in t:
                # Masks are often bool in exported transformer graphs.
                inputs[inp.name] = (np.random.rand(*shape) > 0.5).astype(np.bool_)
            elif "int" in t:
                # Handle int32/int64; keep token id range moderate.
                dtype = np.int64 if "64" in t else np.int32
                inputs[inp.name] = np.random.randint(0, 100, size=shape, dtype=dtype)
            elif "float16" in t:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float16)
            elif "double" in t or "float64" in t:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float64)
            else:
                # Default ONNX float tensor
                inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        outputs = engine.run(inputs)
        
        # 5. Analyze Outputs
        attn_tensors = {k: v for k, v in outputs.items() if exposed_map.get(k) == 'Softmax'}
        act_tensors = {k: v for k, v in outputs.items() if exposed_map.get(k) in ['Relu', 'Gelu', 'Add']}
        
        attn_results = AttentionAnalyzer().analyze(attn_tensors)
        act_results = ActivationAnalyzer().analyze(act_tensors)
        
        # Hallucination (need 3D logits)
        hall_results = {"risk_score": 0, "metrics": {}}
        for v in outputs.values():
            if len(v.shape) == 3:
                hall_results = HallucinationAnalyzer().analyze(v)
                break
        
        # Semantic Analysis (optional — depends on sentence-transformers/TF)
        semantic_results = {}
        if _HAS_SEMANTIC:
            try:
                semantic_analyzer = SemanticAnalyzer()
                semantic_results = semantic_analyzer.analyze(act_tensors, input_text)
            except Exception:
                semantic_results = {"layer_drift": {}}
        
        # 6. Model Health Analysis (new!)
        health_analyzer = ModelHealthAnalyzer()
        health_results = health_analyzer.full_report(attn_tensors, act_tensors)"""

new_onnx_block = "\n".join(["    " + line for line in old_onnx_block.split("\n")])
# We already replaced `1. load model`, so we apply it to everything below
text = text.replace(old_onnx_block, new_onnx_block)


# Add the new plotting functions just above `generate_overview()`
plots_to_add = """
def generate_gradient_plot():
    r = analysis_results.get("gradients", {})
    if not r or "error" in r:
        return None
    stats = r.get("layer_stats", {})
    names = list(stats.keys())
    stds = [stats[n]["std"] for n in names]
    
    fig = go.Figure(data=go.Heatmap(
        z=[stds],
        x=names,
        colorscale='Viridis',
        hoverinfo='x+z'
    ))
    fig.update_layout(title="Gradient Flow Across Layers", height=300)
    return fig

def generate_evolution_plot():
    r = analysis_results.get("evolution", {})
    if not r or "evolution" not in r:
        return None
    
    evo = r["evolution"]
    if not evo: return None
    
    # Pick a few key layers to trace
    keys_to_plot = list(evo.keys())[:5]
    
    fig = go.Figure()
    for k in keys_to_plot:
        fig.add_trace(go.Scatter(
            x=evo[k]["steps"],
            y=evo[k]["l2_dist_from_prev"],
            mode='lines+markers',
            name=k[-20:]
        ))
    fig.update_layout(title="Weight Evolution (L2 Distance over Epochs)", height=400)
    return fig

"""
text = text.replace("def generate_overview():", plots_to_add + "def generate_overview():")


# Modify UI tabs
old_ui_tabs = """            with gr.Tab("Semantic"):
                gr.Markdown("**Semantic Fidelity**", elem_classes="section-title")
                gr.Markdown("Tracking how meaning propagates through the layers (via SBERT)", elem_classes="section-subtitle")
                semantic_plot = gr.Plot()
                with gr.Accordion("Details", open=False):
                    gr.Markdown(XAI_TEXTS["semantic"])

            with gr.Tab("🩺 Model Health"):
                gr.Markdown("**Deep Model Health Diagnostics**", elem_classes="section-title")
                gr.Markdown("Per-head attention entropy, head specialization, and signal propagation analysis", elem_classes="section-subtitle")

                # Summary card
                health_summary_md = gr.Markdown("*Run analysis to see results.*")

                # Heatmap — full width
                health_heatmap = gr.Plot(label="Attention Flow Heatmap")

                # Dashboard + Signal — side by side
                with gr.Row():
                    with gr.Column(scale=1):
                        health_dashboard = gr.Plot(label="Head Specialization")
                    with gr.Column(scale=1):
                        health_signal = gr.Plot(label="Signal Propagation")

                with gr.Accordion("How to read this", open=False):
                    gr.Markdown(XAI_TEXTS["model_health"])

        # Connect
        analyze_btn.click(
            fn=analyze_model,
            inputs=[file_input, text_input],
            outputs=[
                overview_md, weight_plot, attn_plot, act_plot, hall_plot,
                semantic_plot, health_heatmap, health_dashboard,
                health_signal, health_summary_md, status
            ]
        )"""

new_ui_tabs = """            with gr.Tab("Semantic"):
                gr.Markdown("**Semantic Fidelity**", elem_classes="section-title")
                gr.Markdown("Tracking how meaning propagates through the layers (via SBERT)", elem_classes="section-subtitle")
                semantic_plot = gr.Plot()
                with gr.Accordion("Details", open=False):
                    gr.Markdown(XAI_TEXTS["semantic"])

            with gr.Tab("🩺 Model Health"):
                gr.Markdown("**Deep Model Health Diagnostics**", elem_classes="section-title")
                gr.Markdown("Per-head attention entropy, head specialization, and signal propagation analysis", elem_classes="section-subtitle")

                # Summary card
                health_summary_md = gr.Markdown("*Run analysis to see results.*")

                # Heatmap — full width
                health_heatmap = gr.Plot(label="Attention Flow Heatmap")

                # Dashboard + Signal — side by side
                with gr.Row():
                    with gr.Column(scale=1):
                        health_dashboard = gr.Plot(label="Head Specialization")
                    with gr.Column(scale=1):
                        health_signal = gr.Plot(label="Signal Propagation")

                with gr.Accordion("How to read this", open=False):
                    gr.Markdown(XAI_TEXTS["model_health"])
                    
            with gr.Tab("Training Dynamics"):
                gr.Markdown("**Deep Training Metrics (PyTorch Only)**", elem_classes="section-title")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        grad_plot = gr.Plot(label="Gradient Flow")
                        evo_plot = gr.Plot(label="Weight Evolution")
                    with gr.Column(scale=1):
                        gr.Markdown(XAI_TEXTS["gradients"])
                        gr.Markdown(XAI_TEXTS["evolution"])

        # Connect
        analyze_btn.click(
            fn=analyze_model,
            inputs=[file_input, text_input],
            outputs=[
                overview_md, weight_plot, attn_plot, act_plot, hall_plot,
                semantic_plot, health_heatmap, health_dashboard,
                health_signal, health_summary_md, evo_plot, grad_plot, status
            ]
        )"""

text = text.replace(old_ui_tabs, new_ui_tabs)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated app.py successfully.")
