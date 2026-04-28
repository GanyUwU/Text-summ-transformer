"""
Interactive Transformer Visualizer - BERTViz Style

Generates an interactive HTML visualization that shows:
- All layers and all attention heads
- Step-by-step token generation
- Click to explore attention patterns
- Side-by-side comparison of heads

Usage:
    python visualize_attention.py
    
This creates an HTML file you can open in your browser.
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
import sys
import webbrowser

sys.path.insert(0, str(Path(__file__).parent))

from model import build_transformer
from dataset_summarization import SummarizationDataset, causal_mask
from config_summarization import get_config, latest_weights_file_path
from tokenizers import Tokenizer
from datasets import load_dataset


def extract_all_attention(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device):
    """Extract attention weights from ALL layers and heads."""
    model.eval()
    attention_data = {
        'encoder_self': [],      # Encoder self-attention
        'decoder_self': [],      # Decoder self-attention  
        'decoder_cross': [],     # Cross-attention (decoder → encoder)
    }
    
    hooks = []
    
    # Hook for encoder self-attention
    for i, layer in enumerate(model.encoder.layers):
        def make_hook(layer_idx, attn_type):
            def hook(module, input, output):
                if hasattr(module, 'attention_scores'):
                    attn = module.attention_scores.detach().cpu().numpy()
                    attention_data[attn_type].append({
                        'layer': layer_idx,
                        'attention': attn[0].tolist()  # Remove batch dim
                    })
            return hook
        hooks.append(layer.self_attention_block.register_forward_hook(
            make_hook(i, 'encoder_self')
        ))
    
    # Hooks for decoder
    for i, layer in enumerate(model.decoder.layers):
        def make_hook(layer_idx, attn_type):
            def hook(module, input, output):
                if hasattr(module, 'attention_scores'):
                    attn = module.attention_scores.detach().cpu().numpy()
                    attention_data[attn_type].append({
                        'layer': layer_idx,
                        'attention': attn[0].tolist()
                    })
            return hook
        hooks.append(layer.self_attention_block.register_forward_hook(
            make_hook(i, 'decoder_self')
        ))
        hooks.append(layer.cross_attention_block.register_forward_hook(
            make_hook(i, 'decoder_cross')
        ))
    
    # Forward pass
    with torch.no_grad():
        encoder_output = model.encode(encoder_input.to(device), encoder_mask.to(device))
        model.decode(encoder_output, encoder_mask.to(device), 
                    decoder_input.to(device), decoder_mask.to(device))
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return attention_data


def generate_html(attention_data, src_tokens, tgt_tokens, output_path):
    """Generate interactive HTML visualization."""
    
    num_layers = len(attention_data['decoder_cross'])
    num_heads = len(attention_data['decoder_cross'][0]['attention']) if attention_data['decoder_cross'] else 8
    
    html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Transformer Attention Visualizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { opacity: 0.9; }
        
        .container { padding: 20px; max-width: 1400px; margin: 0 auto; }
        
        .controls {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        .control-group label { font-size: 12px; color: #888; text-transform: uppercase; }
        .control-group select, .control-group input {
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid #333;
            background: #0f0f23;
            color: #fff;
            font-size: 14px;
        }
        
        .explanation {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }
        .explanation h3 { color: #667eea; margin-bottom: 10px; }
        .explanation p { line-height: 1.6; color: #aaa; }
        
        .viz-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .panel {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
        }
        .panel h3 { margin-bottom: 15px; color: #667eea; }
        
        .tokens {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 15px;
        }
        .token {
            padding: 5px 10px;
            background: #0f0f23;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
        }
        .token:hover { background: #667eea; }
        .token.selected { background: #764ba2; }
        .token.highlighted { 
            box-shadow: 0 0 0 2px #667eea;
            transform: scale(1.1);
        }
        
        .heatmap-container {
            overflow-x: auto;
        }
        .heatmap {
            display: grid;
            gap: 2px;
            margin-top: 10px;
        }
        .heatmap-cell {
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            border-radius: 3px;
            cursor: pointer;
            transition: transform 0.1s;
        }
        .heatmap-cell:hover { transform: scale(1.2); z-index: 10; }
        
        .head-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .head-mini {
            background: #0f0f23;
            padding: 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .head-mini:hover { background: #1a1a3e; transform: scale(1.02); }
        .head-mini.selected { border: 2px solid #667eea; }
        .head-mini h4 { font-size: 12px; color: #888; margin-bottom: 5px; }
        .mini-heatmap { display: grid; gap: 1px; }
        .mini-cell { width: 8px; height: 8px; border-radius: 1px; }
        
        .step-navigator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 20px;
        }
        .step-btn {
            padding: 10px 20px;
            background: #667eea;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }
        .step-btn:hover { background: #764ba2; }
        .step-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .step-info {
            background: #0f0f23;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .step-info h4 { color: #667eea; margin-bottom: 10px; }
        
        .legend {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-top: 10px;
            font-size: 12px;
        }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 20px; height: 12px; border-radius: 2px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Transformer Attention Visualizer</h1>
        <p>Interactive exploration of attention patterns across layers and heads</p>
    </div>
    
    <div class="container">
        <div class="controls">
            <div class="control-group">
                <label>Attention Type</label>
                <select id="attnType" onchange="updateVisualization()">
                    <option value="decoder_cross">Cross-Attention (Decoder → Encoder)</option>
                    <option value="encoder_self">Encoder Self-Attention</option>
                    <option value="decoder_self">Decoder Self-Attention</option>
                </select>
            </div>
            <div class="control-group">
                <label>Layer</label>
                <select id="layerSelect" onchange="updateVisualization()">
                    ''' + ''.join([f'<option value="{i}">Layer {i}</option>' for i in range(num_layers)]) + '''
                </select>
            </div>
            <div class="control-group">
                <label>Head</label>
                <select id="headSelect" onchange="updateVisualization()">
                    ''' + ''.join([f'<option value="{i}">Head {i}</option>' for i in range(num_heads)]) + '''
                </select>
            </div>
        </div>
        
        <div class="explanation" id="explanation">
            <h3>📚 What am I looking at?</h3>
            <p id="explainText">
                <strong>Cross-Attention</strong> shows how the decoder (generating the summary) 
                attends to the encoder (the input article). Each cell shows how much attention 
                a summary token pays to an article token. Brighter = more attention.
            </p>
        </div>
        
        <div class="viz-container">
            <div class="panel">
                <h3>📰 Input Tokens (Article)</h3>
                <div class="tokens" id="srcTokens"></div>
                
                <h3 style="margin-top: 20px;">📝 Output Tokens (Summary)</h3>
                <div class="tokens" id="tgtTokens"></div>
            </div>
            
            <div class="panel">
                <h3>🎯 Attention Heatmap</h3>
                <div class="heatmap-container">
                    <div class="heatmap" id="heatmap"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #0f0f23;"></div>
                        <span>Low attention</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #667eea;"></div>
                        <span>Medium</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff6b6b;"></div>
                        <span>High attention</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="panel" style="margin-top: 20px;">
            <h3>🧠 All Heads at a Glance (Layer <span id="currentLayer">0</span>)</h3>
            <p style="color: #888; font-size: 13px; margin-bottom: 10px;">
                Click on a head to select it. Different heads learn different patterns!
            </p>
            <div class="head-grid" id="headGrid"></div>
        </div>
        
        <div class="panel" style="margin-top: 20px;">
            <h3>🚶 Step-by-Step Generation</h3>
            <p style="color: #888; font-size: 13px;">
                See how attention changes as the model generates each token.
            </p>
            <div class="step-navigator">
                <button class="step-btn" id="prevBtn" onclick="prevStep()">← Previous</button>
                <span id="stepCounter">Step 1 / ''' + str(len(tgt_tokens)) + '''</span>
                <button class="step-btn" id="nextBtn" onclick="nextStep()">Next →</button>
            </div>
            <div class="step-info" id="stepInfo">
                <h4>Generating token: "<span id="currentToken"></span>"</h4>
                <p id="stepExplanation"></p>
            </div>
        </div>
    </div>
    
    <script>
        // Data
        const attentionData = ''' + json.dumps(attention_data) + ''';
        const srcTokens = ''' + json.dumps(src_tokens) + ''';
        const tgtTokens = ''' + json.dumps(tgt_tokens) + ''';
        
        let currentStep = 0;
        let selectedSrcToken = -1;
        let selectedTgtToken = -1;
        
        // Initialize
        function init() {
            renderTokens();
            updateVisualization();
            updateStep();
        }
        
        function renderTokens() {
            const srcContainer = document.getElementById('srcTokens');
            const tgtContainer = document.getElementById('tgtTokens');
            
            srcContainer.innerHTML = srcTokens.slice(0, 30).map((t, i) => 
                `<span class="token" onclick="selectSrcToken(${i})" id="src-${i}">${t}</span>`
            ).join('');
            
            tgtContainer.innerHTML = tgtTokens.slice(0, 20).map((t, i) => 
                `<span class="token" onclick="selectTgtToken(${i})" id="tgt-${i}">${t}</span>`
            ).join('');
        }
        
        function selectSrcToken(idx) {
            document.querySelectorAll('#srcTokens .token').forEach(t => t.classList.remove('selected'));
            document.getElementById(`src-${idx}`).classList.add('selected');
            selectedSrcToken = idx;
            highlightColumn(idx);
        }
        
        function selectTgtToken(idx) {
            document.querySelectorAll('#tgtTokens .token').forEach(t => t.classList.remove('selected'));
            document.getElementById(`tgt-${idx}`).classList.add('selected');
            selectedTgtToken = idx;
            currentStep = idx;
            updateStep();
            highlightRow(idx);
        }
        
        function highlightColumn(col) {
            // Highlight which tokens attend to this input token
            const attnType = document.getElementById('attnType').value;
            const layer = parseInt(document.getElementById('layerSelect').value);
            const head = parseInt(document.getElementById('headSelect').value);
            
            const layerData = attentionData[attnType].find(d => d.layer === layer);
            if (!layerData) return;
            
            const attn = layerData.attention[head];
            
            document.querySelectorAll('#tgtTokens .token').forEach((t, i) => {
                t.classList.remove('highlighted');
                if (i < attn.length && col < attn[i].length && attn[i][col] > 0.1) {
                    t.classList.add('highlighted');
                }
            });
        }
        
        function highlightRow(row) {
            // Highlight which tokens this output attends to
            const attnType = document.getElementById('attnType').value;
            const layer = parseInt(document.getElementById('layerSelect').value);
            const head = parseInt(document.getElementById('headSelect').value);
            
            const layerData = attentionData[attnType].find(d => d.layer === layer);
            if (!layerData) return;
            
            const attn = layerData.attention[head];
            if (row >= attn.length) return;
            
            document.querySelectorAll('#srcTokens .token').forEach((t, i) => {
                t.classList.remove('highlighted');
                if (i < attn[row].length && attn[row][i] > 0.1) {
                    t.classList.add('highlighted');
                }
            });
        }
        
        function updateVisualization() {
            const attnType = document.getElementById('attnType').value;
            const layer = parseInt(document.getElementById('layerSelect').value);
            const head = parseInt(document.getElementById('headSelect').value);
            
            document.getElementById('currentLayer').textContent = layer;
            
            // Update explanation
            const explanations = {
                'decoder_cross': 'Cross-Attention shows how the decoder looks at the encoder when generating each output token. High attention means the model is focusing on that input word.',
                'encoder_self': 'Encoder Self-Attention shows how each input token relates to other input tokens. This helps the model understand context.',
                'decoder_self': 'Decoder Self-Attention shows how each output token relates to previously generated tokens. This maintains coherence.'
            };
            document.getElementById('explainText').innerHTML = `<strong>${attnType.replace('_', ' ').toUpperCase()}</strong>: ${explanations[attnType]}`;
            
            // Get attention data
            const layerData = attentionData[attnType].find(d => d.layer === layer);
            if (!layerData) {
                document.getElementById('heatmap').innerHTML = '<p>No data for this configuration</p>';
                return;
            }
            
            const attn = layerData.attention[head];
            renderHeatmap(attn);
            renderHeadGrid(layer, attnType);
        }
        
        function renderHeatmap(attn) {
            const container = document.getElementById('heatmap');
            const numRows = Math.min(attn.length, 20);
            const numCols = Math.min(attn[0] ? attn[0].length : 0, 30);
            
            container.style.gridTemplateColumns = `repeat(${numCols}, 30px)`;
            
            let html = '';
            for (let i = 0; i < numRows; i++) {
                for (let j = 0; j < numCols; j++) {
                    const val = attn[i][j];
                    const color = getColor(val);
                    html += `<div class="heatmap-cell" style="background: ${color};" 
                             title="[${tgtTokens[i] || i}] → [${srcTokens[j] || j}]: ${(val*100).toFixed(1)}%"
                             onclick="selectTgtToken(${i}); selectSrcToken(${j});">
                             </div>`;
                }
            }
            container.innerHTML = html;
        }
        
        function renderHeadGrid(layer, attnType) {
            const container = document.getElementById('headGrid');
            const layerData = attentionData[attnType].find(d => d.layer === layer);
            if (!layerData) return;
            
            const currentHead = parseInt(document.getElementById('headSelect').value);
            const numHeads = layerData.attention.length;
            
            let html = '';
            for (let h = 0; h < numHeads; h++) {
                const attn = layerData.attention[h];
                const selected = h === currentHead ? 'selected' : '';
                
                html += `<div class="head-mini ${selected}" onclick="document.getElementById('headSelect').value=${h}; updateVisualization();">
                    <h4>Head ${h}</h4>
                    <div class="mini-heatmap" style="grid-template-columns: repeat(10, 8px);">`;
                
                // Render mini heatmap (10x10 max)
                for (let i = 0; i < Math.min(10, attn.length); i++) {
                    for (let j = 0; j < Math.min(10, attn[i].length); j++) {
                        const color = getColor(attn[i][j]);
                        html += `<div class="mini-cell" style="background: ${color};"></div>`;
                    }
                }
                html += '</div></div>';
            }
            container.innerHTML = html;
        }
        
        function getColor(val) {
            // Blue to red gradient
            const intensity = Math.min(1, val * 2);
            if (intensity < 0.5) {
                const g = Math.round(intensity * 2 * 126);
                return `rgb(${15 + g}, ${15 + g}, ${35 + intensity * 200})`;
            } else {
                const r = Math.round((intensity - 0.5) * 2 * 255);
                return `rgb(${102 + r * 0.6}, ${126 - r * 0.3}, ${234 - r * 0.5})`;
            }
        }
        
        function prevStep() {
            if (currentStep > 0) {
                currentStep--;
                updateStep();
            }
        }
        
        function nextStep() {
            if (currentStep < tgtTokens.length - 1) {
                currentStep++;
                updateStep();
            }
        }
        
        function updateStep() {
            document.getElementById('stepCounter').textContent = `Step ${currentStep + 1} / ${tgtTokens.length}`;
            document.getElementById('prevBtn').disabled = currentStep === 0;
            document.getElementById('nextBtn').disabled = currentStep >= tgtTokens.length - 1;
            
            const token = tgtTokens[currentStep] || '';
            document.getElementById('currentToken').textContent = token;
            
            // Find top attended input tokens
            const attnType = document.getElementById('attnType').value;
            const layer = parseInt(document.getElementById('layerSelect').value);
            const head = parseInt(document.getElementById('headSelect').value);
            
            const layerData = attentionData[attnType].find(d => d.layer === layer);
            if (layerData && currentStep < layerData.attention[head].length) {
                const attn = layerData.attention[head][currentStep];
                const sorted = attn.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v).slice(0, 3);
                
                const topWords = sorted.map(x => `"${srcTokens[x.i] || '?'}" (${(x.v*100).toFixed(0)}%)`).join(', ');
                document.getElementById('stepExplanation').innerHTML = 
                    `When generating "<strong>${token}</strong>", the model focused most on: ${topWords}`;
            }
            
            // Highlight current token
            selectTgtToken(currentStep);
        }
        
        init();
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return output_path


def main():
    print("\n" + "="*60)
    print("🔍 INTERACTIVE TRANSFORMER VISUALIZER")
    print("="*60)
    
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format('shared'))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Vocabulary: {tokenizer.get_vocab_size()}")
    
    # Build model
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=config['src_seq_len'],
        tgt_seq_len=config['tgt_seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)
    
    # Load weights
    weights_path = latest_weights_file_path(config)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded: {weights_path}")
    
    # Load sample
    print("\nLoading sample data...")
    # ds = load_dataset(config['datasource'], config['dataset_version'], split='validation[:1]')
    # dataset = SummarizationDataset(ds, tokenizer, config['src_seq_len'], config['tgt_seq_len'])
    # sample = dataset[0]
    
    # Replace loading sample from dataset with custom text
    article = "YOUR ARTICLE TEXT HERE"
    summary = "YOUR REFERENCE SUMMARY (optional)"
    
    enc_ids = tokenizer.encode(article)[:config['src_seq_len']-2]
    dec_ids = tokenizer.encode(summary)[:config['tgt_seq_len']-2] if summary else [tokenizer.bos_id]

    sample = {
        'encoder_input': torch.tensor([ [tokenizer.bos_id] + enc_ids + [tokenizer.eos_id] + [tokenizer.pad_id] * (config['src_seq_len'] - (len(enc_ids)+2)) ], dtype=torch.long)[0],
        'decoder_input': torch.tensor([ [tokenizer.bos_id] + dec_ids + [tokenizer.pad_id] * (config['tgt_seq_len'] - (len(dec_ids)+1)) ], dtype=torch.long)[0],
        'encoder_mask': (torch.tensor([ [tokenizer.bos_id] + enc_ids + [tokenizer.eos_id] ] ) != tokenizer.pad_id).unsqueeze(0).unsqueeze(0),
        'decoder_mask': torch.tril(torch.ones((1, config['tgt_seq_len'], config['tgt_seq_len']), dtype=torch.bool)),
        'src_text': article,
        'tgt_text': summary or ''
    }
    
    # Add batch dimension
    encoder_input = sample['encoder_input'].unsqueeze(0)
    decoder_input = sample['decoder_input'].unsqueeze(0)
    encoder_mask = sample['encoder_mask'].unsqueeze(0)
    decoder_mask = sample['decoder_mask'].unsqueeze(0)
    
    print("\n📰 Article (first 150 chars):")
    print(f"   {sample['src_text'][:150]}...")
    print(f"\n📝 Summary:")
    print(f"   {sample['tgt_text']}")
    
    # Extract attention
    print("\n🔍 Extracting attention patterns...")
    attention_data = extract_all_attention(
        model, encoder_input, encoder_mask, decoder_input, decoder_mask, device
    )
    
    print(f"   Encoder self-attention: {len(attention_data['encoder_self'])} layers")
    print(f"   Decoder self-attention: {len(attention_data['decoder_self'])} layers")
    print(f"   Cross-attention: {len(attention_data['decoder_cross'])} layers")
    
    # Get tokens for display
    enc_ids = sample['encoder_input'].tolist()
    dec_ids = sample['decoder_input'].tolist()
    
    src_tokens = [tokenizer.decode([t]).strip()[:12] for t in enc_ids[:50]]
    tgt_tokens = [tokenizer.decode([t]).strip()[:12] for t in dec_ids[:30]]
    
    # Generate HTML
    print("\n🎨 Generating interactive visualization...")
    output_path = Path('attention_visualizer.html')
    generate_html(attention_data, src_tokens, tgt_tokens, output_path)
    
    print(f"\n✅ Visualization saved to: {output_path.absolute()}")
    print("\n📖 Opening in browser...")
    
    # Open in browser
    webbrowser.open(f'file://{output_path.absolute()}')
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Select attention type (cross, encoder-self, decoder-self)")
    print("2. Choose layer and head to explore")
    print("3. Click on tokens to see attention patterns")
    print("4. Use step-by-step to see generation process")
    print("5. Explore different heads - they learn different patterns!")


if __name__ == '__main__':
    main()
