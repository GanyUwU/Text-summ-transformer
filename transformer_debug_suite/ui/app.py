"""
Transformer Debug Suite - Interactive Gradio UI

Run with: python -m transformer_debug_suite.ui.app
Opens in browser at http://localhost:7860
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
import sys
import tempfile

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from debug_suite.core.model_loader import ModelLoader
from debug_suite.core.graph_instrumenter import GraphInstrumenter
from debug_suite.core.inference_engine import InferenceEngine
from debug_suite.analyzers.weights import WeightAnalyzer
from debug_suite.analyzers.attention import AttentionAnalyzer
from debug_suite.analyzers.activations import ActivationAnalyzer
from debug_suite.analyzers.hallucination import HallucinationAnalyzer
try:
    from debug_suite.analyzers.semantic import SemanticAnalyzer
    _HAS_SEMANTIC = True
except Exception:
    _HAS_SEMANTIC = False
from debug_suite.analyzers.model_health import ModelHealthAnalyzer
from debug_suite.analyzers.weight_evolution import WeightEvolutionAnalyzer
from debug_suite.analyzers.gradient_flow import GradientFlowAnalyzer
import torch

try:
    from model import build_transformer
    import pretrain_config
    _HAS_TORCH_MODEL = True
except ImportError:
    _HAS_TORCH_MODEL = False

from debug_suite.analyzers.weight_evolution import WeightEvolutionAnalyzer
from debug_suite.analyzers.gradient_flow import GradientFlowAnalyzer
import torch

try:
    from model import build_transformer
    import pretrain_config
    _HAS_TORCH_MODEL = True
except ImportError:
    _HAS_TORCH_MODEL = False

from debug_suite.analyzers.weight_evolution import WeightEvolutionAnalyzer
from debug_suite.analyzers.gradient_flow import GradientFlowAnalyzer
import torch

try:
    from model import build_transformer
    from pretrain_config import get_finetune_config
    _HAS_TORCH_MODEL = True
except ImportError:
    _HAS_TORCH_MODEL = False

# ==================== ENHANCED THEME CONFIG ====================
SLATE_PALETTE = {
    "50": "#f8fafc",
    "100": "#f1f5f9",
    "200": "#e2e8f0",
    "300": "#cbd5e1",
    "400": "#94a3b8",
    "500": "#64748b",
    "600": "#475569",
    "700": "#1a1a1a",
    "800": "#0a0a0a",
    "900": "#000000",
    "950": "#020617",
}

ACCENT_PALETTE = {
    "primary": "#6366f1",      # Indigo-500
    "primary_hover": "#4f46e5", # Indigo-600
    "secondary": "#22d3ee",     # Cyan-400
    "success": "#10b981",      # Emerald-500
    "warning": "#f59e0b",       # Amber-500
    "danger": "#ef4444",       # Red-500
}

# Unified Plotly template - register as a proper template
_debug_suite_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#000000",
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        color="#cbd5e1",
        size=12,
    ),
    title=dict(
        font=dict(size=16, color="#f1f5f9", family="Inter"),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.1)",
        linecolor="#1a1a1a",
        tickcolor="#64748b",
        ticks="outside",
        title=dict(font=dict(color="#94a3b8")),
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.1)",
        linecolor="#1a1a1a",
        tickcolor="#64748b",
        ticks="outside",
        title=dict(font=dict(color="#94a3b8")),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#1a1a1a",
        borderwidth=1,
        font=dict(color="#94a3b8"),
    ),
    coloraxis=dict(
        colorbar=dict(
            bgcolor="#0a0a0a",
            bordercolor="#1a1a1a",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#cbd5e1"),
        )
    ),
    margin=dict(l=60, r=30, t=60, b=60),
)

pio.templates["debug_suite"] = go.layout.Template(layout=_debug_suite_layout)
pio.templates.default = "plotly_dark+debug_suite"

ENTROPY_COLORSCALE = [
    [0.0, "#dc2626"],   # Red - collapsed
    [0.25, "#f97316"],  # Orange
    [0.5, "#eab308"],   # Yellow
    [0.75, "#22c55e"],  # Green
    [1.0, "#15803d"],   # Dark green
]

HEALTH_COLORSCALE = [
    [0.0, "#991b1b"],   # Dark red
    [0.5, "#f59e0b"],   # Amber  
    [1.0, "#059669"],   # Emerald
]

DIVERGING_COLORSCALE = [
    [0.0, "#ef4444"],
    [0.25, "#f97316"],
    [0.5, "#eab308"],
    [0.75, "#22c55e"],
    [1.0, "#10b981"],
]

# ==================== XAI EXPLANATIONS ====================
XAI_TEXTS = {
    "overview": """
##  What is this tool?
This debug suite analyzes your ONNX Transformer model to identify potential issues:
- **Dead neurons** that don't contribute to output
- **Collapsed attention heads** that focus on single tokens
- **Vanishing signals** where information gets lost
- **Hallucination risk** based on output uncertainty

Upload your model above and click Analyze to begin.
""",
    "weights": """
##  Weight Analysis Explained

**What we measure:**
- **Mean/Std**: Distribution of weight values per layer
- **Dead Neurons**: Weights with magnitude < 1e-5 (contribute nothing)

**Why it matters:**
Dead neurons indicate:
1. Poor initialization
2. Vanishing gradients during training
3. Over-pruning

**Recommendation:** If >10% neurons are dead, consider retraining with better initialization.
""",
    "attention": """
##  Attention Analysis Explained

**Entropy** measures how "spread out" attention is:
- **Low entropy (< 0.3)**: Head focuses on 1-2 tokens → May be too specialized
- **High entropy (> 0.9)**: Head attends uniformly → May be redundant
- **Medium entropy**: Healthy selective attention

**Head Importance** ranking helps identify which heads matter most.

**Recommendation:** Heads with very low or very high entropy may be candidates for pruning.
""",
    "activations": """
##  Activation Analysis Explained

**Sparsity** = percentage of zero/near-zero activations
- High sparsity (>90%) after ReLU = "dying ReLU" problem
- Signal may not propagate through network

**Variance Flow** shows if signal strength is preserved:
- Decreasing variance = vanishing signal
- Exploding variance = unstable training

**Recommendation:** If variance drops >100x from input to output, model has vanishing gradient issues.
""",
    "hallucination": """
##  Hallucination Risk Explained

**Risk Score (0-100)** combines:
1. **Output Entropy**: High = model unsure what to predict
2. **Confidence**: Low max probability = uncertain
3. **Repetition**: Repeated n-grams = degenerate generation

**Recommendation:** High-risk models need more training data or better decoding strategies.
""",
    "semantic": """
##  Semantic Fidelity Explained

**What we measure:**
- **Semantic Drift**: How much the representation changes between consecutive layers.
- **Meaning Persistence**: If the "core concept" (e.g., the subject of the sentence) is preserved or getting scrambled.

**Why it matters:**
In a healthy Transformer, the meaning should refine across layers. A sudden drop in similarity indicates a layer that is "breaking" the representation.
""",

    "gradients": """
## Gradient Flow Explained

**What we measure:**
- **Simulated Backward Pass**: We run synthetic data through the model to check backpropagation.
- **Vanishing Gradients**: Standard deviation < 1e-6. Signal dies before reaching early layers.
- **Exploding Gradients**: Standard deviation > 10.0. Training becomes unstable.

**How to fix:**
- **Vanishing**: Check for dying ReLUs or apply Xavier/Kaiming initialization. Remove excessive LayerNorms.
- **Exploding**: Apply gradient clipping (`clip_grad_norm_`) or reduce learning rate.
""",
    "evolution": """
## Weight Evolution Explained

**What we measure:**
- **L2 Distance over Epochs**: How much the weights are actually moving during training.

**How to fix:**
- **Frozen Encoder**: If distance < 1e-4, the learning rate is too low or gradients are vanishing.
- **Erratic Movement**: If distance spikes wildly, lower learning rate or increase batch size.
""",

    "gradients": """
## Gradient Flow Explained

**What we measure:**
- **Simulated Backward Pass**: We run synthetic data through the model to check backpropagation.
- **Vanishing Gradients**: Standard deviation < 1e-6. Signal dies before reaching early layers.
- **Exploding Gradients**: Standard deviation > 10.0. Training becomes unstable.

**How to fix:**
- **Vanishing**: Check for dying ReLUs or apply Xavier/Kaiming initialization. Remove excessive LayerNorms.
- **Exploding**: Apply gradient clipping (`clip_grad_norm_`) or reduce learning rate.
""",
    "evolution": """
## Weight Evolution Explained

**What we measure:**
- **L2 Distance over Epochs**: How much the weights are actually moving during training.

**How to fix:**
- **Frozen Encoder**: If distance < 1e-4, the learning rate is too low or gradients are vanishing.
- **Erratic Movement**: If distance spikes wildly, lower learning rate or increase batch size.
""",
    "model_health": """
## 🩺 Model Health Explained

**Attention Flow Heatmap** shows normalized entropy per attention head across all layers:
- 🟢 **Green (0.6–0.8)**: Healthy — head is specialized (local, global, or sparse pattern)
- 🟡 **Yellow (>0.8)**: Uniform — head distributes attention equally (no specialization)
- 🔴 **Red (<0.25)**: Collapsed — head fixates on one or two tokens (broken)

**Head Specialization** classifies each head as:
- **Local**: Attends to nearby tokens (sliding window pattern)
- **Global**: Attends broadly across the sequence
- **Sparse**: Sharp attention on specific important tokens
- **Uniform**: No learned pattern (usually early in training)
- **Collapsed**: Stuck on one token (needs reinitialization or entropy regularization)

**Signal Propagation** tracks representation std through layers:
- Stable std = healthy gradient flow
- Growing >3x between layers = explosion risk
- Shrinking <0.1x = vanishing signal
"""
}

# ==================== GLOBAL STATE ====================
analysis_results = {}
current_model_path = None

# ==================== ANALYSIS FUNCTIONS ====================

def analyze_model(file, input_text):
    """Main analysis function called when user uploads and clicks Analyze."""
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
                    d_model=config['d_model'], N=config['num_layers'], h=config['num_heads'],
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
            loader = ModelLoader(current_model_path)
            model = loader.load()
            metadata = loader.get_metadata()
        
            # 2. Weight Analysis (Static)
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
            health_results = health_analyzer.full_report(attn_tensors, act_tensors)
        
        # Store results
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
                health_signal, health_summary, health_summary, evo_plot, grad_plot, "✅ Analysis complete!")
        
    except Exception as e:
        import traceback
        return (f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```",) + (None,) * 13


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



def generate_overview():
    """Generate overview markdown with improved styling."""
    r = analysis_results
    if not r:
        return "No analysis yet."
    
    # Calculate health score
    health = 100
    issues = []
    
    # Weight issues
    dead = r["weights"]["summary"].get("dead_neurons", 0)
    exploding = r["weights"]["summary"].get("exploding_layers", 0)
    if dead > 0:
        health -= min(30, dead * 3)
        issues.append(f"🔴 **Dead Neurons**: {dead} detected - potential vanishing gradients")
    if exploding > 0:
        health -= min(40, exploding * 10)
        issues.append(f"🔴 **Exploding Weights**: {exploding} layers - potential instability")
    
    # Attention issues
    for issue in r["attention"].get("head_issues", []):
        sev = issue.get("severity", "warning")
        prefix = "🔴" if sev == "critical" else "🟡"
        health -= 5 if sev == "warning" else 10
        issues.append(f"{prefix} {issue['msg']}")
    
    # Semantic Drift
    drift_issues = r.get("semantic", {}).get("issues", [])
    for issue in drift_issues:
        health -= 10
        issues.append(f"🟡 {issue['msg']}")

    # Hallucination
    risk = r["hallucination"].get("risk_score", 0)
    if risk > 60:
        health -= 20
        issues.append(f"🔴 **Critical Hallucination Risk**: {risk:.1f}/100")
    elif risk > 30:
        health -= 10
        issues.append(f"🟡 **Moderate Hallucination Risk**: {risk:.1f}/100")
    
    health = max(0, health)
    
    # Determine status color
    if health >= 80:
        status_emoji = "✅"
        status_color = "#10b981"
        status_text = "Excellent"
    elif health >= 50:
        status_emoji = "⚠️"
        status_color = "#f59e0b"
        status_text = "Fair"
    else:
        status_emoji = "❌"
        status_color = "#ef4444"
        status_text = "Poor"
    
    # Format output with styled markdown
    md = f"""
## 🏥 Model Health Overview

| Metric | Value |
|--------|-------|
| **Health Score** | <span style="color:{status_color}; font-size:1.5rem; font-weight:bold;">{health}/100</span> {status_emoji} |
| **Status** | {status_text} |
| **Parameters** | {r['weights']['summary'].get('total_params', 'N/A'):,} |
| **Inputs** | {', '.join(r['metadata'].get('inputs', ['N/A']))} |
| **Outputs** | {', '.join(r['metadata'].get('outputs', ['N/A']))} |

"""
    
    if issues:
        md += "## 🚨 Issues Detected\n\n"
        for issue in issues:
            md += f"- {issue}\n"
    else:
        md += "## ✅ No Major Issues Detected\n\nYour model appears to be healthy! All metrics are within acceptable ranges.\n"
    
    return md

def generate_weight_plot():
    """Generate weight distribution plot."""
    r = analysis_results.get("weights", {})
    if not r:
        return None
    
    layer_stats = r.get("layer_stats", {})
    if not layer_stats:
        return None
    
    names = list(layer_stats.keys())[:20]  # Limit to 20 layers
    means = [layer_stats[n]["mean"] for n in names]
    stds = [layer_stats[n]["std"] for n in names]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Mean', x=names, y=means))
    fig.add_trace(go.Bar(name='Std', x=names, y=stds))
    fig.update_layout(
        title="Weight Statistics per Layer",
        barmode='group',
        xaxis_title="Layer",
        yaxis_title="Value",
        height=550
    )
    return fig

def generate_attention_plot():
    """Generate attention analysis visualization with enhanced styling."""
    r = analysis_results.get("attention", {})
    head_stats = r.get("head_stats", {})
    
    # If we have runtime attention data, show entropy heatmap
    if head_stats:
        data = []
        labels = []
        for layer_name, heads in head_stats.items():
            labels.append(layer_name[:30])
            row = [h.get("normalized_entropy", 0) for h in heads]
            data.append(row)
        
        if data:
            fig = px.imshow(
                data,
                labels=dict(x="Head", y="Layer", color="Entropy"),
                y=labels,
                aspect="auto",
                color_continuous_scale=ENTROPY_COLORSCALE,
                title="👁️ Attention Entropy Heatmap"
            )
            fig.update_layout(
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.3)",
                font=dict(color="#cbd5e1"),
                title=dict(font=dict(size=16, color="#f1f5f9")),
                xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
                yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
            )
            return fig
    
    # Fallback: Analyze attention weights from model (QKV matrices)
    w_results = analysis_results.get("weights", {})
    layer_stats = w_results.get("layer_stats", {})
    
    # Find attention-related weights (query, key, value, attention)
    attn_layers = {}
    for name, stats in layer_stats.items():
        name_lower = name.lower()
        if any(k in name_lower for k in ['query', 'key', 'value', 'attention', 'q_proj', 'k_proj', 'v_proj']):
            # Group by layer number
            import re
            match = re.search(r'layers?[._]?(\d+)', name_lower)
            layer_num = int(match.group(1)) if match else 0
            
            if layer_num not in attn_layers:
                attn_layers[layer_num] = []
            attn_layers[layer_num].append({
                "name": name[:40],
                "std": stats.get("std", 0),
                "mean": abs(stats.get("mean", 0))
            })
    
    if not attn_layers:
        # No attention weights found - show info message
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No attention patterns available</b><br><br>" +
                 "This model may use fused attention operations<br>" +
                 "that don't expose intermediate Softmax outputs.<br><br>" +
                 "Check the Weights tab for attention layer analysis.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            title="👁️ Attention Analysis",
            height=450,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.3)",
            font=dict(color="#cbd5e1"),
        )
        return fig
    
    # Create bar chart of attention weight statistics per layer
    layers = sorted(attn_layers.keys())
    avg_stds = []
    for l in layers:
        avg_std = np.mean([w["std"] for w in attn_layers[l]])
        avg_stds.append(avg_std)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"L{l}" for l in layers],
        y=avg_stds,
        marker_color=px.colors.sequential.Viridis
    ))
    fig.update_layout(
        title="Attention Weight Variance per Layer (from QKV projections)",
        xaxis_title="Layer",
        yaxis_title="Average Std Dev",
        height=550
    )
    return fig

def generate_activation_plot():
    """Generate activation sparsity/variance plot."""
    r = analysis_results.get("activations", {})
    layer_stats = r.get("layer_stats", {})
    
    if not layer_stats:
        return None
    
    names = list(layer_stats.keys())[:15]
    sparsity = [layer_stats[n]["sparsity"] * 100 for n in names]
    variance = [min(layer_stats[n]["variance"], 10) for n in names]  # Cap at 10
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names, y=sparsity, mode='lines+markers', name='Sparsity %'))
    fig.add_trace(go.Scatter(x=names, y=variance, mode='lines+markers', name='Variance', yaxis='y2'))
    
    fig.update_layout(
        title="Activation Analysis",
        yaxis=dict(title="Sparsity %"),
        yaxis2=dict(title="Variance", overlaying='y', side='right'),
        height=550
    )
    return fig

def generate_hallucination_plot():
    """Generate hallucination risk gauge with enhanced styling."""
    r = analysis_results.get("hallucination", {})
    risk = r.get("risk_score", 0)
    
    # Determine color based on risk level
    if risk >= 60:
        risk_color = "#ef4444"  # Red
        risk_emoji = "🔴"
        risk_text = "HIGH RISK"
    elif risk >= 30:
        risk_color = "#f59e0b"  # Amber
        risk_emoji = "🟡"
        risk_text = "MODERATE"
    else:
        risk_color = "#10b981"  # Green
        risk_emoji = "✅"
        risk_text = "LOW RISK"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={
            'text': f"🎭 Hallucination Risk — {risk_emoji} {risk_text}",
            'font': {'size': 16, 'color': '#f1f5f9'}
        },
        number={
            'suffix': '/100', 
            'font': {'size': 36, 'color': risk_color, 'family': 'JetBrains Mono'}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'dtick': 20},
            'bar': {'color': risk_color, 'thickness': 0.4},
            'bgcolor': '#0a0a0a',
            'bordercolor': '#1a1a1a',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.8,
                'value': risk,
            },
        }
    ))
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig

def generate_semantic_plot():
    """Generate semantic drift plot with -2 to 2 range."""
    r = analysis_results.get("semantic", {})
    drift = r.get("layer_drift", {})
    
    if not drift:
        return None
        
    layers = list(drift.keys())
    similarities = [drift[l] for l in layers]
    
    # Create diverging colorscale based on values
    colors = []
    for val in similarities:
        if val >= 0:
            colors.append("#22c55e")  # Green for positive
        else:
            colors.append("#ef4444")  # Red for negative
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=similarities,
        mode='lines+markers',
        name='Semantic Drift',
        line=dict(color='#a855f7', width=3),
        marker=dict(size=10, color=colors, line=dict(color='#1e1e1e', width=2)),
        hovertemplate="<b>%{x}</b><br>Drift: %{y:.4f}<extra></extra>",
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#64748b", line_width=1)
    
    fig.update_layout(
        title=dict(text="🧠 Semantic Drift — Layer-to-Layer", font=dict(size=16, color="#f1f5f9")),
        xaxis_title="<b>Layer Transition</b>",
        yaxis_title="<b>Drift Score</b>",
        yaxis=dict(range=[-2, 2], gridcolor="rgba(148, 163, 184, 0.1)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ==================== MODEL HEALTH PLOTS ====================

def generate_health_heatmap():
    """Attention Flow Heatmap — entropy per head per layer with enhanced styling."""
    health = analysis_results.get("health", {})
    attn_h = health.get("attention_health", {})
    heatmap = attn_h.get("heatmap_data", [])
    types_grid = attn_h.get("heatmap_types", [])
    layers_info = attn_h.get("layers", [])

    if not heatmap:
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No attention data available</b><br><br>Run analysis to see the attention flow heatmap",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, 
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            height=400, 
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#000000"
        )
        return fig

    num_heads = len(heatmap[0]) if heatmap else 0
    layer_labels = [f"L{i}" for i in range(len(heatmap))]
    head_labels = [f"H{i}" for i in range(num_heads)]

    # Build enhanced hover text
    hover = []
    for li, row in enumerate(heatmap):
        hover_row = []
        for hi, val in enumerate(row):
            t = types_grid[li][hi] if li < len(types_grid) and hi < len(types_grid[li]) else "?"
            color_map = {
                "collapsed": "#ef4444",
                "uniform": "#f59e0b", 
                "sparse": "#22c55e",
                "local": "#22c55e",
                "global": "#22d3ee"
            }
            color = color_map.get(t, "#94a3b8")
            hover_row.append(
                f"<b style='color:{color}'>●</b> <b>Layer {li}, Head {hi}</b><br>" +
                f"Entropy: <b>{val:.1%}</b><br>" +
                f"Type: <span style='color:{color}'>{t.upper()}</span>"
            )
        hover.append(hover_row)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=head_labels,
        y=layer_labels,
        colorscale=ENTROPY_COLORSCALE,
        zmin=0, zmax=1,
        hovertext=hover,
        hoverinfo="text",
        colorbar=dict(
            title=dict(text="Entropy", font=dict(color="#cbd5e1")),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Collapsed", "Sparse", "Moderate", "Global", "Uniform"],
            tickfont=dict(color="#94a3b8"),
            bgcolor="#0a0a0a",
            bordercolor="#1a1a1a",
        ),
        hovertemplate="%{hovertext}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="🧠 Attention Flow — Per-Head Entropy Heatmap",
            font=dict(size=16, color="#f1f5f9")
        ),
        xaxis_title="<b>Attention Head</b>",
        yaxis_title="<b>Layer</b>",
        yaxis=dict(autorange="reversed", gridcolor="rgba(148, 163, 184, 0.1)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        height=max(350, 50 * len(heatmap) + 120),
        margin=dict(l=60, r=30, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
    )
    return fig


def generate_health_dashboard():
    """Head specialization donut + health gauge with enhanced styling."""
    from plotly.subplots import make_subplots

    health = analysis_results.get("health", {})
    attn_h = health.get("attention_health", {})
    summary = attn_h.get("summary", {})
    types = summary.get("head_types", {})
    score = attn_h.get("health_score", 0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "indicator"}]],
        column_widths=[0.55, 0.45],
    )

    # Donut chart — head type distribution with enhanced colors
    labels = []
    values = []
    colors = []
    color_map = {
        "local": "#22c55e",     # Green
        "global": "#22d3ee",    # Cyan
        "sparse": "#a855f7",    # Purple
        "uniform": "#f59e0b",   # Amber
        "collapsed": "#ef4444", # Red
    }
    for k, v in types.items():
        if v > 0:
            labels.append(k.capitalize())
            values.append(v)
            colors.append(color_map.get(k, "#64748b"))

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=11, color="#cbd5e1"),
            hoverinfo="label+value+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ),
        row=1, col=1
    )

    # Health gauge with enhanced styling
    if score >= 75:
        gauge_color = "#10b981"  # Emerald
    elif score >= 50:
        gauge_color = "#f59e0b"  # Amber
    else:
        gauge_color = "#ef4444"  # Red
        
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "🩺 Health Score", "font": {"size": 14, "color": "#f1f5f9"}},
            number={"suffix": "/100", "font": {"size": 32, "color": gauge_color, "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b", "dtick": 25},
                "bar": {"color": gauge_color, "thickness": 0.4},
                "bgcolor": "#0a0a0a",
                "bordercolor": "#1a1a1a",
                "borderwidth": 2,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        ),
        row=1, col=2
    )

    total = summary.get("total_heads", 0)
    healthy = summary.get("healthy_heads", 0)
    fig.update_layout(
        title=dict(
            text=f"🎯 Head Specialization — {healthy}/{total} Healthy",
            font=dict(size=16, color="#f1f5f9")
        ),
        height=380,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def generate_health_signal():
    """Signal propagation — std across layers with enhanced styling."""
    health = analysis_results.get("health", {})
    signal = health.get("signal_propagation", {})
    std_vals = signal.get("std_values", [])
    names = signal.get("layer_names", [])

    if not std_vals:
        return None

    # Shorten names for readability
    short = [n[:20] + "…" if len(n) > 20 else n for n in names]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=short, y=std_vals,
        mode="lines+markers",
        name="Activation Std",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=8, color="#6366f1"),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.15)",
        hovertemplate="<b>%{x}</b><br>Std: %{y:.4f}<extra></extra>",
    ))

    # Add danger zone thresholds
    if std_vals:
        max_std = max(std_vals)
        fig.add_hline(
            y=max_std * 3, 
            line_dash="dash",
            line_color="rgba(239, 68, 68, 0.6)",
            annotation_text="⚠️ Explosion",
            annotation_position="top right",
            annotation=dict(font=dict(color="#ef4444"))
        )
        
        # Vanishing threshold
        if max_std > 0:
            fig.add_hline(
                y=max_std * 0.1, 
                line_dash="dot",
                line_color="rgba(245, 158, 11, 0.6)",
                annotation_text="⚠️ Vanishing",
                annotation_position="bottom right",
                annotation=dict(font=dict(color="#f59e0b"))
            )

    fig.update_layout(
        title=dict(
            text="📈 Signal Propagation — Activation Std",
            font=dict(size=16, color="#f1f5f9")
        ),
        xaxis_title="<b>Layer</b>",
        yaxis_title="<b>Std Dev</b>",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(tickangle=-45, gridcolor="rgba(148, 163, 184, 0.1)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        margin=dict(l=60, r=30, t=60, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    return fig


def generate_health_summary_md():
    """Markdown summary with alerts."""
    health = analysis_results.get("health", {})
    attn_h = health.get("attention_health", {})
    signal = health.get("signal_propagation", {})

    score = attn_h.get("health_score", 0)
    summary = attn_h.get("summary", {})
    alerts = attn_h.get("alerts", []) + signal.get("growth_alerts", [])
    types = summary.get("head_types", {})

    grade = ("🟢 EXCELLENT" if score >= 90
             else "🟢 GOOD" if score >= 75
             else "🟡 FAIR" if score >= 50
             else "🔴 POOR")

    md = f"## Overall: {grade} ({score}/100)\n\n"
    md += f"**Total heads:** {summary.get('total_heads', 0)} | "
    md += f"**Healthy:** {summary.get('healthy_heads', 0)}\n\n"

    md += "| Type | Count | Status |\n|---|---|---|\n"
    status_map = {
        "local": "✅ Healthy", "global": "✅ Healthy", "sparse": "✅ Healthy",
        "uniform": "⚠️ Under-trained", "collapsed": "❌ Broken"
    }
    for t in ["local", "global", "sparse", "uniform", "collapsed"]:
        c = types.get(t, 0)
        if c > 0:
            md += f"| {t.capitalize()} | {c} | {status_map[t]} |\n"

    if alerts:
        md += "\n### ⚡ Alerts\n\n"
        for a in alerts[:10]:
            md += f"- {a}\n"
    else:
        md += "\n### ✅ No critical issues detected.\n"

    return md


# ==================== GRADIO UI ==

CUSTOM_CSS = """
/* Sleek Dark Theme - Enhanced */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Header Styling */
.header-section {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(34, 211, 238, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.header-section h1 {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    letter-spacing: -0.02em;
}

.header-section p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    margin-top: 0.5rem !important;
}

/* Health Score Banner */
.health-banner {
    background: linear-gradient(135deg, #0a0a0a 0%, #000000 100%);
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.health-score {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.health-label {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #0a0a0a 0%, #000000 100%);
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
}

.metric-card .title {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.metric-card .subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Status Colors */
.status-good { color: #10b981 !important; }
.status-warning { color: #f59e0b !important; }
.status-danger { color: #ef4444 !important; }
.status-info { color: #22d3ee !important; }

/* Issue Badges */
.issue-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
}

.issue-badge.critical {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}

.issue-badge.warning {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
}

.issue-badge.info {
    background: rgba(34, 211, 238, 0.15);
    border: 1px solid rgba(34, 211, 238, 0.3);
    color: #a5f3fc;
}

/* Tab Styling */
.tabs-container button[role="tab"] {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #64748b !important;
    padding: 0.75rem 1rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}

.tabs-container button[role="tab"]:hover {
    color: #cbd5e1 !important;
    background: rgba(99, 102, 241, 0.1) !important;
}

.tabs-container button[role="tab"][aria-selected="true"] {
    color: #f1f5f9 !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

/* Section Headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1rem;
}

/* Cards/Panels */
.gr-group, .gr-panel {
    background: linear-gradient(135deg, #0a0a0a 0%, #000000 100%) !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
}

/* Input Styling */
.gr-input {
    background: #000000 !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

.gr-input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* Button Styling */
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

/* File Upload */
.gr-file {
    background: #000000 !important;
    border: 2px dashed #1a1a1a !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

.gr-file:hover {
    border-color: #6366f1 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* Accordion */
.gr-accordion {
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    overflow: hidden;
}

.gr-accordion-details {
    background: #000000 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #000000;
}

::-webkit-scrollbar-thumb {
    background: #1a1a1a;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* Plot Container */
.plot-container {
    background: #000000;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 1rem;
}

/* Loading Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-pulse {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Toast/Notification */
.toast-success {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
    padding: 0.75rem 1rem;
    border-radius: 8px;
}

.toast-warning {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
    padding: 0.75rem 1rem;
    border-radius: 8px;
}

.toast-error {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
    padding: 0.75rem 1rem;
    border-radius: 8px;
}

/* Responsive */
@media (max-width: 768px) {
    .health-banner {
        flex-direction: column;
        text-align: center;
    }
    
    .header-section {
        padding: 1rem;
    }
}
"""

def create_ui():
    with gr.Blocks(
        title="Transformer Debug Suite",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="cyan",
        ).set(
            body_background_fill="#000000",
            body_background_fill_dark="#000000",
            block_background_fill="#0a0a0a",
            block_background_fill_dark="#0a0a0a",
            block_border_color="#1a1a1a",
            block_label_text_color="#666666",
            block_title_text_color="#e0e0e0",
            input_background_fill="#0a0a0a",
            input_background_fill_dark="#0a0a0a",
            button_primary_background_fill="#6366f1",
            button_primary_background_fill_hover="#4f46e5",
            button_secondary_background_fill="#1a1a1a",
            button_secondary_background_fill_hover="#2a2a2a",
            button_cancel_background_fill="#ef4444",
            button_cancel_background_fill_hover="#dc2626",
        ),
        css=CUSTOM_CSS
    ) as app:
        
        # Header with gradient
        with gr.Group(elem_classes="header-section"):
            gr.Markdown("# Transformer Debug Suite")
            gr.Markdown("Analyze ONNX transformer models for potential issues and get actionable insights")
        
        # Input Section - Clean layout
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, variant="compact"):
                file_input = gr.File(
                    label="📦 Model File",
                    file_types=[".onnx"],
                    height=100
                )
            with gr.Column(scale=2, variant="compact"):
                text_input = gr.Textbox(
                    label="🔍 Test Prompt",
                    placeholder="Enter a sentence to probe the model's behavior...",
                    lines=2
                )
        
        # Analyze Button
        with gr.Row():
            analyze_btn = gr.Button("⚡ Run Full Diagnostic Suite", variant="primary", size="lg")
        
        # Health Score Banner (hidden until analysis)
        with gr.Group(elem_classes="health-banner", visible=False) as health_banner:
            health_score_display = gr.Markdown("**Health Score: --**")
            health_issues_display = gr.Markdown("*Run analysis to see health metrics*")
        
        # Status
        status = gr.Markdown("✓ Ready - Upload a model to begin", elem_classes="status-text")
        
        # Main Tabs
        with gr.Tabs(elem_classes="tabs-container"):
            
            with gr.Tab("📊 Overview"):
                overview_md = gr.Markdown(XAI_TEXTS["overview"])
                health_summary_md = gr.Markdown("*Run analysis to see health summary*")
            
            with gr.Tab("⚖️ Weights"):
                gr.Markdown("### Weight Distribution", elem_classes="section-title")
                gr.Markdown("Statistical distribution of model weights per layer", elem_classes="section-subtitle")
                weight_plot = gr.Plot()
                with gr.Accordion("📋 What does this mean?", open=False):
                    gr.Markdown(XAI_TEXTS["weights"])
            
            with gr.Tab("👁️ Attention"):
                gr.Markdown("### Attention Patterns", elem_classes="section-title")
                gr.Markdown("Entropy and variance of attention patterns across heads", elem_classes="section-subtitle")
                attn_plot = gr.Plot()
                with gr.Accordion("📋 What does this mean?", open=False):
                    gr.Markdown(XAI_TEXTS["attention"])
            
            with gr.Tab("📈 Activations"):
                gr.Markdown("### Signal Flow", elem_classes="section-title")
                gr.Markdown("Information propagation through the network", elem_classes="section-subtitle")
                act_plot = gr.Plot()
                with gr.Accordion("📋 What does this mean?", open=False):
                    gr.Markdown(XAI_TEXTS["activations"])
            
            with gr.Tab("🎭 Hallucination"):
                gr.Markdown("### Output Reliability", elem_classes="section-title")
                gr.Markdown("Estimated likelihood of unreliable outputs", elem_classes="section-subtitle")
                hall_plot = gr.Plot()
                with gr.Accordion("📋 What does this mean?", open=False):
                    gr.Markdown(XAI_TEXTS["hallucination"])
            
            with gr.Tab("🧠 Semantic"):
                gr.Markdown("### Semantic Fidelity", elem_classes="section-title")
                gr.Markdown("Tracking how meaning propagates through the layers", elem_classes="section-subtitle")
                semantic_plot = gr.Plot()
                with gr.Accordion("Details", open=False):
                    gr.Markdown(XAI_TEXTS["semantic"])

            with gr.Tab("🩺 Model Health"):
                gr.Markdown("### Deep Model Health Diagnostics", elem_classes="section-title")
                gr.Markdown("Per-head attention entropy, head specialization, and signal propagation", elem_classes="section-subtitle")

                # Summary
                health_summary_md2 = gr.Markdown("*Run analysis to see results.*")

                # Heatmap - Full width
                health_heatmap = gr.Plot(label="Attention Flow Heatmap")

                # Dashboard + Signal - side by side
                with gr.Row():
                    with gr.Column():
                        health_dashboard = gr.Plot(label="Head Specialization")
                    with gr.Column():
                        health_signal = gr.Plot(label="Signal Propagation")

                with gr.Accordion("📖 How to read this", open=False):
                    gr.Markdown(XAI_TEXTS["model_health"])
                    
            with gr.Tab("📉 Training Dynamics"):
                gr.Markdown("### Deep Training Metrics (PyTorch Only)", elem_classes="section-title")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        grad_plot = gr.Plot(label="Gradient Flow")
                        evo_plot = gr.Plot(label="Weight Evolution")
                    with gr.Column(scale=1):
                        gr.Markdown("### 📖 Understanding These Metrics", elem_classes="section-title")
                        with gr.Accordion("Gradient Flow", open=True):
                            gr.Markdown(XAI_TEXTS["gradients"])
                        with gr.Accordion("Weight Evolution", open=True):
                            gr.Markdown(XAI_TEXTS["evolution"])

        # Connect
        analyze_btn.click(
            fn=analyze_model,
            inputs=[file_input, text_input],
            outputs=[
                overview_md, weight_plot, attn_plot, act_plot, hall_plot,
                semantic_plot, health_heatmap, health_dashboard,
                health_signal, health_summary_md, health_summary_md2, evo_plot, grad_plot, status
            ]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)


