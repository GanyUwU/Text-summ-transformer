import gradio as gr
import torch
import torch.nn.functional as F
from inference import load_model, summarize, _causal_mask
from pretrain_config import get_finetune_config

# --- Load Model & Config (Cached) ---
def get_model():
    """Load model once and cache it."""
    try:
        config = get_finetune_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {device}...")
        model, tokenizer = load_model(config, device)
        return model, tokenizer, config, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

MODEL, TOKENIZER, CONFIG, DEVICE = get_model()

# --- Prediction Function ---
def generate_summary(article_text, max_length=150, min_length=30, ngram_size=3, rep_penalty=1.2, temp=1.0, beam_size=4, length_penalty=0.6):
    if MODEL is None:
        return "Error: Model could not be loaded. Please check if checkpoints exist."
    
    if len(article_text.strip()) < 50:
        return "Error: Article is too short. Please enter at least 50 characters."

    try:
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
        return summary
    except Exception as e:
        return f"Error during generation: {str(e)}"

# --- Gradio UI ---
custom_css = """
.container { max_width: 900px; margin: auto; }
.output-text textarea { font-size: 16px; line-height: 1.5; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📝 Transformer Summarization Demo
        
        This interface runs the custom **Pointer-Generator Transformer** trained on CNN/DailyMail.
        
        **Features:**
        - **Copy Mechanism:** Can copy rare words (names, dates) from source.
        - **Weight Sharing:** Efficient ~60M parameter model.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Article",
                placeholder="Paste your article here...",
                lines=15,
                max_lines=30
            )
            
            with gr.Accordion("Advanced Settings", open=True):
                max_len_slider = gr.Slider(
                    minimum=50, maximum=300, value=128, step=1, 
                    label="Max Summary Length"
                )
                min_len_slider = gr.Slider(
                    minimum=10, maximum=100, value=30, step=1, 
                    label="Min Summary Length (Force longer output)"
                )
                ngram_slider = gr.Slider(
                    minimum=0, maximum=5, value=3, step=1,
                    label="No Repeat N-Gram Size (0 to disable)"
                )
                rep_penalty_slider = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                    label="Repetition Penalty (1.0 = none)"
                )
                temp_slider = gr.Slider(
                    minimum=0.5, maximum=1.5, value=1.0, step=0.1,
                    label="Temperature (Generativity)"
                )
                beam_slider = gr.Slider(
                    minimum=1, maximum=10, value=4, step=1,
                    label="Beam Size (1 = Greedy, >1 = Search)"
                )
                penalty_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.6, step=0.1,
                    label="Beam Length Penalty"
                )
            
            submit_btn = gr.Button("Generate Summary", variant="primary", size="lg")
            
            # Example articles
            gr.Examples(
                examples=[
                    ["Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The fish, which lives at depths of over 8,000 meters, has unique adaptations that allow it to survive the extreme pressure. Researchers from the University of Tokyo used remotely operated vehicles to capture footage of the creature. The discovery adds to our understanding of life in the deepest parts of the ocean."],
                    ["The local council has approved plans for a new community park in the city center. Construction is set to begin next month and is expected to take approximately one year to complete. The park will feature a large playground, walking trails, and a dedicated area for community events. Residents have expressed excitement about the project, noting the need for more green spaces in the urban area."],
                ],
                inputs=input_text
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Summary",
                lines=10,
                interactive=False,
                elem_classes="output-text"
            )

    # Event handlers
    submit_btn.click(
        fn=generate_summary, 
        inputs=[
            input_text, max_len_slider, min_len_slider, 
            ngram_slider, rep_penalty_slider, temp_slider,
            beam_slider, penalty_slider
        ], 
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(share=False)
