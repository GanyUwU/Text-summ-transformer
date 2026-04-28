import gradio as gr
import torch
import math
import time
from pathlib import Path
import torch.nn.functional as F
from inference import load_model, summarize, _causal_mask
from checkpoint_utils import load_checkpoint
from pretrain_config import get_finetune_config

# --- Load Model & Config (Cached) ---
def get_model():
    """Load model once and cache it."""
    try:
        config = get_finetune_config()

        # Override the loading folder for inference explicitly to use Phase 8 weights
        config['model_folder'] = 'weights_v11_nuclear'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {device}...")
        model, tokenizer = load_model(config, device)

        # If a specific step checkpoint is required, attempt to load and override.
        specific_ckpt = Path('weights_v11_nuclear/nuclear_summarizer_best.pt')
        if specific_ckpt.exists():
            try:
                ck = load_checkpoint(str(specific_ckpt), map_location=device)
                if 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'])
                    print(f"Overrode loaded weights with {specific_ckpt}")
                else:
                    print(f"Checkpoint {specific_ckpt} missing 'model_state_dict'; skipping")
            except Exception as e:
                print(f"Failed to load specific checkpoint {specific_ckpt}: {e}")

        return model, tokenizer, config, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

MODEL, TOKENIZER, CONFIG, DEVICE = get_model()

# --- Prediction Function (backend logic preserved exactly) ---
def generate_summary(article_text, max_length=150, min_length=30, ngram_size=3, rep_penalty=1.2, temp=1.0, beam_size=4, length_penalty=0.6):
    if MODEL is None:
        return "Error: Model could not be loaded. Please check if checkpoints exist.", ""

    if len(article_text.strip()) < 50:
        return "Error: Article is too short. Please enter at least 50 characters.", ""

    try:
        start_time = time.time()
        # Generate summary
        summary = summarize(
            MODEL, TOKENIZER, article_text, CONFIG, DEVICE,
            max_length=int(max_length), min_length=int(min_length),
            no_repeat_ngram_size=int(ngram_size),
            repetition_penalty=float(rep_penalty),
            temperature=float(temp),
            beam_size=int(beam_size),
            length_penalty=float(length_penalty)
        )
        elapsed = time.time() - start_time

        # Compute statistics
        input_words = len(article_text.split())
        output_words = len(summary.split())
        compression = round(output_words / max(input_words, 1) * 100, 1)
        reading_time = max(1, round(output_words / 238))  # avg reading speed

        stats_html = f"""
        <div class="stats-row">
            <div class="stat-chip">
                <span class="stat-label">Words</span>
                <span class="stat-value">{output_words}</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-chip">
                <span class="stat-label">Compression</span>
                <span class="stat-value">{compression}%</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-chip">
                <span class="stat-label">Read time</span>
                <span class="stat-value">~{reading_time} min</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-chip">
                <span class="stat-label">Time</span>
                <span class="stat-value">{elapsed:.1f}s</span>
            </div>
        </div>
        """
        return summary, stats_html
    except Exception as e:
        return f"Error during generation: {str(e)}", ""


def count_input_stats(text):
    """Live word/token count for input box."""
    if not text or not text.strip():
        return '<span class="input-counter">0 words · ~0 tokens</span>'
    words = len(text.split())
    # Rough subword estimate: ~1.3 tokens per word for SentencePiece
    tokens_est = int(words * 1.3)
    color = "var(--text-muted)"
    return f'<span class="input-counter" style="color:{color}">{words} words · ~{tokens_est} tokens</span>'


# ==================== CUSTOM CSS ====================
custom_css = """
/* ===== Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== CSS Reset & Base (Premium Matte SaaS Theme) ===== */
:root {
    --bg-page: #f9f9fb; /* Light mode default fallback */
    --bg-card: #ffffff;
    --border-subtle: #e5e7eb;
    --border-focus: #c7d2fe;
    
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --text-muted: #9ca3af;
    
    --accent-primary: #4f46e5;
    --accent-hover: #4338ca;
    
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-focus: 0 0 0 3px rgba(79, 70, 229, 0.15);
    
    --transition: 0.15s ease-in-out;
}

/* Force Dark Mode Overrides for Premium Look */
body.dark, .dark {
    --bg-page: #0f1115;      /* Arc-style deep charcoal */
    --bg-card: #181a1f;      /* Matte elevated surface */
    --bg-input: #1f2228;     /* Slightly lighter for inputs */
    
    --border-subtle: #2d3139; /* Notion-style subtle border */
    --border-focus: #4f46e5;
    
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.4);
    --shadow-md: 0 8px 16px -4px rgba(0,0,0,0.5);
    --shadow-focus: 0 0 0 2px rgba(79, 70, 229, 0.4);
}

/* ===== Global Override ===== */
.gradio-container {
    background: var(--bg-page) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 32px 24px !important;
}

.main, .contain { background: transparent !important; }
footer { display: none !important; }

/* ===== Hero Header ===== */
.header-wrapper {
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-subtle);
}

.header-left {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.product-title {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}

.product-subtitle {
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}

.product-badge {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: var(--shadow-sm);
}

/* ===== Layout Structure ===== */
.workspace-grid {
    gap: 20px !important;
}

/* ===== Panel Cards ===== */
.panel-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 0 !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden;
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-card);
}

.panel-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
}

.panel-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-left: auto;
}

.panel-body {
    padding: 16px;
}

/* ===== Textbox Overrides (Editor Feel) ===== */
.gradio-container textarea,
.gradio-container .input-text textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: inherit !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 16px !important;
    transition: all var(--transition) !important;
    resize: vertical !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.1) !important;
}

.gradio-container textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: var(--shadow-focus) !important;
    outline: none !important;
}

.gradio-container textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Output Panel Style */
.output-card textarea {
    background: var(--bg-card) !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 4px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

.output-card textarea:focus {
    box-shadow: none !important;
}

/* ===== Live Counter ===== */
.input-counter {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
}

.counter-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 4px 4px;
}

/* ===== Primary CTA Button (Stripe/Linear style) ===== */
.generate-btn button {
    width: auto !important;
    padding: 8px 16px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    background: var(--accent-primary) !important;
    color: white !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    cursor: pointer !important;
    transition: all var(--transition) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}

.generate-btn button:hover {
    background: var(--accent-hover) !important;
    transform: none !important;
}

.generate-btn button:active {
    background: #3730a3 !important;
}

/* ===== Statistics Row ===== */
.stats-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-input);
    border-top: 1px solid var(--border-subtle);
    font-family: 'Inter', sans-serif;
}

.stat-chip {
    display: flex;
    align-items: baseline;
    gap: 6px;
}

.stat-divider {
    width: 1px;
    height: 12px;
    background: var(--border-subtle);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
}

.stat-value {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-primary);
}

/* ===== Accordion / Settings ===== */
.gradio-container .gradio-accordion {
    background: transparent !important;
    border: none !important;
    margin-top: 16px !important;
}

.gradio-container .gradio-accordion > .label-wrap {
    background: transparent !important;
    padding: 8px 0 !important;
    border: none !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

.gradio-container .gradio-accordion > .label-wrap span {
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}

.settings-group-label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    margin-top: 16px !important;
    margin-bottom: 8px !important;
    padding-bottom: 4px !important;
    border-bottom: 1px dashed var(--border-subtle) !important;
}

/* Label text overrides */
.gradio-container label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ===== Slider Overrides ===== */
.gradio-container input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

.gradio-container .gradio-slider input[type="number"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
}

/* ===== Examples ===== */
.gradio-container .gradio-examples {
    border: none !important;
    margin-top: 12px !important;
}

.gradio-container .gradio-examples .gallery-item {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    transition: all var(--transition) !important;
    padding: 10px 12px !important;
    box-shadow: var(--shadow-sm) !important;
}

.gradio-container .gradio-examples .gallery-item:hover {
    border-color: var(--border-focus) !important;
}

/* Remove default gradio padding in forms */
.gradio-container .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.gradio-container .form {
    background: transparent !important;
    border: none !important;
}
"""


# ==================== EXAMPLES ====================
EXAMPLE_ARTICLES = [
    ["Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The fish, which lives at depths of over 8,000 meters, has unique adaptations that allow it to survive the extreme pressure. Researchers from the University of Tokyo used remotely operated vehicles to capture footage of the creature. The discovery adds to our understanding of life in the deepest parts of the ocean. The team published their findings in the journal Nature, noting that the fish has a unique gelatinous body structure that helps it withstand pressures exceeding 800 atmospheres."],
    ["The local council has approved plans for a new community park in the city center. Construction is set to begin next month and is expected to take approximately one year to complete. The park will feature a large playground, walking trails, and a dedicated area for community events. Residents have expressed excitement about the project, noting the need for more green spaces in the urban area. The project is being funded through a combination of federal grants and local tax revenues, with a total budget of approximately $4.5 million."],
    ["A major breakthrough in renewable energy storage was announced today by researchers at MIT. The team has developed a new type of solid-state battery that can store three times more energy than current lithium-ion batteries while being significantly safer and cheaper to produce. The technology uses abundant materials like iron and sulfur, eliminating the need for expensive cobalt and nickel. Industry experts predict this could accelerate the transition to electric vehicles and grid-scale renewable energy storage within the next five years."],
]


# ==================== BUILD UI ====================
with gr.Blocks(
    css=custom_css,
    title="TRACEUM Summarizer",
) as demo:

    # ===== HEADER =====
    gr.HTML("""
    <div class="header-wrapper">
        <div class="header-left">
            <div class="product-title">TRACEUM Summarizer</div>
            <div class="product-subtitle">Document summarization powered by Pointer-Generator Transformer</div>
        </div>
        <div class="product-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            Model Ready
        </div>
    </div>
    """)

    # ===== MAIN LAYOUT =====
    with gr.Row(elem_classes="workspace-grid"):

        # ===== LEFT COLUMN: INPUT =====
        with gr.Column(scale=5):
            
            gr.HTML('<div class="panel-card"><div class="panel-header"><span class="panel-title">Source Material</span></div><div class="panel-body">')

            input_text = gr.Textbox(
                label="",
                placeholder="Paste your document here to generate a summary...",
                lines=12,
                max_lines=24,
                show_label=False,
                elem_classes="input-text",
            )
            
            with gr.Row(elem_classes="counter-wrapper"):
                input_stats = gr.HTML(
                    value='<span class="input-counter">0 words</span>'
                )
                submit_btn = gr.Button(
                    "Generate Summary",
                    variant="primary",
                    elem_classes="generate-btn",
                )
            
            gr.HTML('</div></div>') # End panel-body & panel-card

            # Live counter update
            input_text.change(
                fn=count_input_stats,
                inputs=input_text,
                outputs=input_stats,
                show_progress="hidden",
            )

            # ===== ADVANCED SETTINGS =====
            with gr.Accordion("Settings", open=False):
                # --- Length Controls ---
                gr.HTML('<div class="settings-group-label">Constraints</div>')
                max_len_slider = gr.Slider(
                    minimum=50, maximum=300, value=150, step=10,
                    label="Max Length",
                    info="Maximum output tokens"
                )
                min_len_slider = gr.Slider(
                    minimum=10, maximum=100, value=30, step=1,
                    label="Min Length",
                    info="Minimum output tokens"
                )

                # --- Diversity Controls ---
                gr.HTML('<div class="settings-group-label">Generation</div>')
                ngram_slider = gr.Slider(
                    minimum=0, maximum=5, value=3, step=1,
                    label="Block N-Grams",
                    info="Size of repeated n-grams to block (0 to disable)"
                )
                rep_penalty_slider = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                    label="Repetition Penalty",
                    info="Penalty for recycling tokens"
                )
                temp_slider = gr.Slider(
                    minimum=0.5, maximum=1.5, value=1.0, step=0.1,
                    label="Temperature",
                    info="Output distribution scaling"
                )

                # --- Decoding Controls ---
                gr.HTML('<div class="settings-group-label">Decoding</div>')
                beam_slider = gr.Slider(
                    minimum=1, maximum=10, value=4, step=1,
                    label="Beam Size",
                    info="Search horizon width"
                )
                penalty_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.6, step=0.1,
                    label="Length Penalty",
                    info="Encourage vs penalize longer beam paths"
                )

            # ===== EXAMPLES =====
            gr.Examples(
                examples=EXAMPLE_ARTICLES,
                inputs=input_text,
                label="Quick Examples",
            )

        # ===== RIGHT COLUMN: OUTPUT =====
        with gr.Column(scale=5):
            
            gr.HTML('<div class="panel-card"><div class="panel-header"><span class="panel-title">Summary</span><span class="panel-subtitle">Output</span></div><div class="panel-body">')

            output_text = gr.Textbox(
                label="",
                lines=14,
                max_lines=24,
                interactive=False,
                show_label=False,
                elem_classes="output-card",
                placeholder="Generated summary will appear here.",
                show_copy_button=True,
            )
            
            gr.HTML('</div>') # End panel-body
            stats_display = gr.HTML(value="")
            gr.HTML('</div>') # End panel-card


    # ===== EVENT HANDLERS =====
    submit_btn.click(
        fn=generate_summary,
        inputs=[
            input_text, max_len_slider, min_len_slider,
            ngram_slider, rep_penalty_slider, temp_slider,
            beam_slider, penalty_slider
        ],
        outputs=[output_text, stats_display],
    )


# ==================== LAUNCH ====================
if __name__ == "__main__":
    demo.launch(share=False)
