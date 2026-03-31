import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    # def forward(self, x):
    #     # x: (batch, seq_len, hidden_size)
    #      # Keep the dimension for broadcasting
    #     mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
    #     # Keep the dimension for broadcasting
    #     std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
    #     # eps is to prevent dividing by zero or when std is very small
    #     return self.alpha * (x - mean) / (std + self.eps) + self.bias
    def forward(self, x):
        # x: (B, T, D)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)   # stable population variance
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * xhat + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        
        # NOTE: Weights are now initialized centrally in initialize_weights()
        # to ensure depth-scaling (1/sqrt(2N)) is protected.

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > self.pe.shape[1]:
            # Dynamically extrapolate positional embeddings on-the-fly if we exceed pre-trained length
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe.requires_grad_(False)
        else:
            x = x + (self.pe[:, :seq_len, :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Stable multi-head attention:
      - uses LayerNorm on Q/K (no extra division by vector-norm)
      - scaled dot-product with sqrt(head_dim)
      - per-head learnable temperature (small scalar) to control sharpness
      - logits clamped to avoid softmax numerical saturation
      - stores attention_probs in `self.attention_scores` (detached) for diagnostics
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # LayerNorm applied to projected q/k (stabilizes logits)
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Per-head learnable temperature (initialized to 1.0)
        # shape: (num_heads, 1, 1) so broadcasting over (B, H, Tq, Tk)
        #self.head_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.logit_temp = nn.Parameter(torch.zeros(self.num_heads, 1, 1))  # trainable
        self.min_temp = 0.1
        self.max_temp = 2.0

        self.logit_clamp_min = -12.0
        self.logit_clamp_max = 12.0

        # placeholder for diagnostics
        self.attention_scores = None

    def forward(self, q, k, v, mask=None, return_attn=False):
        """
        q: (B, Tq, D)
        k: (B, Tk, D)
        v: (B, Tv, D)
        mask: broadcastable mask (B, 1, Tq, Tk) or None
        """
        B, Tq, _ = q.size()
        Tk = k.size(1)

        # 1) Linear projections
        q_proj = self.q_proj(q)    # (B, Tq, D)
        k_proj = self.k_proj(k)    # (B, Tk, D)
        v_proj = self.v_proj(v)    # (B, Tv, D)

        # 2) Stabilize with LayerNorm on projected q/k (do NOT divide by full vector norm)
        q_proj = self.q_norm(q_proj)
        k_proj = self.k_norm(k_proj)

        # 3) Reshape into heads -> (B, H, T, head_dim)
        q_heads = q_proj.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_proj.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_proj.view(B, v_proj.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # 4) Scaled dot-product
        # attn_logits shape: (B, H, Tq, Tk)
        attn_logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) Temperature scaling per head (learned small scalar)
        #attn_logits = attn_logits * self.head_temperature
        temp = torch.nn.functional.softplus(self.logit_temp)   # positive
        temp = torch.clamp(temp, min=self.min_temp, max=self.max_temp)
        attn_logits = attn_logits * temp

        # 6) Safety clamp to keep logits in a numerically sane range
        attn_logits = attn_logits.clamp(min=self.logit_clamp_min, max=self.logit_clamp_max)

        # 7) Apply mask if provided (use large negative for masked positions)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e4)

        # 8) Softmax -> attention probabilities
        attn_probs = torch.softmax(attn_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 9) Context
        context_heads = torch.matmul(attn_probs, v_heads)  # (B, H, Tq, head_dim)

        # 10) Merge heads -> (B, Tq, D)
        # Reverting to explicit Tq to catch dimension logic errors early as requested
        context = context_heads.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        output = self.out_proj(context)

        # 11) Store detached attention probabilities for diagnostics (cheap & safe)
        # detach to avoid extra autograd graph and reduce memory
        self.attention_scores = attn_probs.detach() if attn_probs is not None else None

        if return_attn:
            # avg over heads -> (B, Tq, Tk)
            avg_attn = attn_probs.mean(dim=1) if attn_probs is not None else None
            return output, avg_attn

        return output
# --- End replacement ---

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_cross_attn=False):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        if return_cross_attn:
            # Pre-norm residual: norm FIRST, then pass to attention (matches standard path)
            normed = self.residual_connections[1].norm(x)
            cross_out, cross_attn_weights = self.cross_attention_block(
                normed, encoder_output, encoder_output, src_mask, return_attn=True
            )
            try:
                x = x + self.residual_connections[1].dropout(cross_out)
            except Exception as e:
                import sys
                sys.stderr.write(f"\n[DECODER CROSS RES ERROR] x: {x.shape}, cross_out: {cross_out.shape}\n")
                sys.stderr.flush()
                raise e
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x, cross_attn_weights
        else:
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_cross_attn=False):
        cross_attn = None
        for layer in self.layers:
            if return_cross_attn:
                x, cross_attn = layer(x, encoder_output, src_mask, tgt_mask, return_cross_attn=True)
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)
        # Return the cross-attention from the LAST decoder layer (most relevant for copy)
        if return_cross_attn:
            return self.norm(x), cross_attn
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        # Scale tied logits to prevent explosion (since embeddings are scaled by sqrt(d_model))
        return self.proj(x) / math.sqrt(self.d_model)


class CopyMechanism(nn.Module):
    """
    Pointer-Generator Copy Mechanism (See et al., 2017).
    
    At each decoding step, computes p_gen: the probability of GENERATING
    a word from the vocabulary vs COPYING a word from the source input.
    
    Final distribution = p_gen * vocab_dist + (1 - p_gen) * copy_dist
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # p_gen is computed from: decoder state + context vector + decoder input
        self.w_gen = nn.Linear(d_model * 3, 1)
        
        # FIX: LayerNorm for decoder input embeddings to stop 275x variance explosion
        self.embed_norm = LayerNormalization(d_model)

        # Neutral bias initialization: Let the data and "Detox" drive the balance.
        try:
            nn.init.constant_(self.w_gen.bias, 0.0) 
        except Exception:
            pass
    
    def forward(self, decoder_output, context_vector, decoder_input_embed,
                vocab_logits, cross_attn_weights, src_input_ids, force_no_pointer: bool = False,
                force_all_gen: bool = False):
        """
        Args:
            decoder_output:     (batch, tgt_len, d_model) - decoder hidden state
            context_vector:     (batch, tgt_len, d_model) - weighted sum of encoder states
            decoder_input_embed:(batch, tgt_len, d_model) - decoder input embeddings
            vocab_logits:       (batch, tgt_len, vocab_size) - from projection layer
            cross_attn_weights: (batch, tgt_len, src_len) - attention over source
            src_input_ids:      (batch, src_len) - source token IDs for scatter
            force_no_pointer:   bool - disable pointer (output-only, not used in forward)
            force_all_gen:      bool - HARD DROPOUT: set p_gen = 1.0 for entire batch (forces generation)
        
        Returns:
            final_dist:         (batch, tgt_len, vocab_size) - blended distribution
            p_gen:              (batch, tgt_len, 1) - generation probability
        """
        # Compute generation probability with normalized embedding input
        # This prevents 275x std explosion by bringing embeddings back to unit variance
        normed_embed = self.embed_norm(decoder_input_embed)
        p_gen_input = torch.cat([decoder_output, context_vector, normed_embed], dim=-1)
        
        # Clamp logits to prevent sigmoid saturation (vanishing gradient for gating)
        p_gen_logits = self.w_gen(p_gen_input)
        p_gen_logits = p_gen_logits.clamp(min=-10.0, max=10.0)
        p_gen = torch.sigmoid(p_gen_logits)  # (batch, tgt_len, 1)

        # HARD POINTER DROPOUT: Force p_gen = 1.0 for entire batch (forces pure generation for this step)
        # This is a aggressive regularization that breaks the copying shortcut.
        if force_all_gen:
            p_gen = torch.ones_like(p_gen)
        
        # Vocabulary distribution (from projection layer)
        vocab_dist = torch.softmax(vocab_logits, dim=-1)  # (batch, tgt_len, vocab_size)
        vocab_dist = p_gen * vocab_dist
        
        # Copy distribution (from cross-attention weights)
        copy_dist = (1 - p_gen) * cross_attn_weights  # (batch, tgt_len, src_len)
        
        # Scatter-add copy probabilities onto the vocabulary
        # src_input_ids: (batch, src_len) -> expand to (batch, tgt_len, src_len)
        # Be defensive: ensure we have a 2D tensor of token ids (long) before expanding.
        src_ids = src_input_ids
        if src_ids.dtype == torch.bool:
            raise TypeError(
                "CopyMechanism expected `src_input_ids` (token ids of dtype torch.long), "
                "but got a boolean mask. Pass encoder token ids (shape (batch, src_len)), not encoder mask."
            )
        # Ensure correct dtype and device FIRST
        src_ids = src_ids.long().to(vocab_dist.device)
        # Safe reshape: remove only specific singleton dims (never remove batch dim)
        while src_ids.dim() > 2:
            # Remove dim 1 (the singleton between batch and seq)
            src_ids = src_ids.squeeze(1)
        if src_ids.dim() == 1:
            # Single sample — add batch dim back
            src_ids = src_ids.unsqueeze(0)
        if src_ids.dim() != 2:
            raise ValueError(f"src_input_ids must be 2D (batch, src_len); got shape {tuple(src_ids.shape)}")

        src_ids_expanded = src_ids.unsqueeze(1).expand(-1, copy_dist.size(1), -1)
        
        # Clamp indices to [0, vocab_size - 1] to prevent out-of-bounds scatter_add
        vocab_size = vocab_dist.size(-1)
        src_ids_expanded = src_ids_expanded.clamp(0, vocab_size - 1)
        
        final_dist = vocab_dist.scatter_add(
            dim=-1,
            index=src_ids_expanded,
            src=copy_dist
        )
        
        # Issue #1 Fix: Renormalize to prevent floating point accumulation preventing sum to 1.0
        final_dist = final_dist / final_dist.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        
        return final_dist, p_gen


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer,
                 copy_mechanism: CopyMechanism = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.copy_mechanism = copy_mechanism  # None = no copy (backward compatible)

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask, return_cross_attn=False):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask,
                           return_cross_attn=return_cross_attn)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(self, src, src_mask, tgt, tgt_mask, force_no_pointer: bool = False):
        """Standard forward pass for ONNX export and standard PyTorch calls."""
        return self.forward_with_copy(src, src_mask, tgt, tgt_mask, force_no_pointer)
    
    def forward_with_copy(self, src, src_mask, tgt, tgt_mask, force_no_pointer: bool = False,
                          force_all_gen: bool = False):
        """
        Full forward pass with copy mechanism.
        Used during training when copy_mechanism is enabled.
        
        Returns:
            final_dist:  (batch, tgt_len, vocab_size) - blended generate+copy distribution
            p_gen:       (batch, tgt_len, 1) - generation probability for logging
            cross_attn:  (batch, tgt_len, src_len) - live cross-attention (for coverage loss)
        """
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode with cross-attention weights (live, not detached)
        decoder_output, cross_attn = self.decode(
            encoder_output, src_mask, tgt, tgt_mask, return_cross_attn=True
        )
        
        # Get vocab logits
        vocab_logits = self.project(decoder_output)
        
        if self.copy_mechanism is None:
            return vocab_logits, None, cross_attn
        
        # Get decoder input embeddings (for p_gen computation)
        tgt_embed = self.tgt_embed(tgt)
        tgt_embed = self.tgt_pos(tgt_embed)
        
        # Context vector: weighted sum of encoder output using cross-attention
        # cross_attn: (batch, tgt_len, src_len), encoder_output: (batch, src_len, d_model)
        context_vector = torch.bmm(cross_attn, encoder_output)
        
        # Compute blended distribution
        final_dist, p_gen = self.copy_mechanism(
            decoder_output, context_vector, tgt_embed,
            vocab_logits, cross_attn, src, force_no_pointer,
            force_all_gen=force_all_gen  # Pass through from caller (hard pointer dropout)
        )
        
        return final_dist, p_gen, cross_attn


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int,
                      tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8,
                      dropout: float=0.1, d_ff: int=2048,
                      share_weights: bool=True, use_copy: bool=True) -> Transformer:
    """
    Build transformer with optional weight sharing and copy mechanism.
    
    Args:
        share_weights: If True, share embeddings between src, tgt, and projection.
                       Requires src_vocab_size == tgt_vocab_size.
        use_copy:      If True, add pointer-generator copy mechanism.
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    
    if share_weights and src_vocab_size == tgt_vocab_size:
        # Weight sharing: same embedding for source and target
        tgt_embed = src_embed
    else:
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Weight tying: projection shares weights with decoder target embedding
    if share_weights:
        projection_layer.proj.weight = tgt_embed.embedding.weight
    
    # Create copy mechanism (optional)
    copy_mechanism = CopyMechanism(d_model) if use_copy else None
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,
                              src_pos, tgt_pos, projection_layer, copy_mechanism)
    
    # NOTE: Individual modules (FeedForwardBlock, MultiHeadAttentionBlock) already
    # initialize their own weights in __init__. A blanket xavier_uniform_ here would
    # OVERWRITE the tied embedding<->projection weights, corrupting the shared space.
    
    return transformer


def initialize_weights(model: nn.Module, n_layers: int = 6):
    """
    Standard Xavier uniform initialization with depth-scaling for deep transformers.
    
    Why: In deep residual networks, signal variance grows linearly with depth. 
    Scaling residual-entering projections by 1/sqrt(2N) maintains unit variance 
    through the stack. This fix prevents 'Semantic Melt' in deep models.
    
    Safely handles tied weights (e.g. Embedding <-> Projection) by tracking weight IDs.
    """
    seen_params = set()
    
    for name, module in model.named_modules():
        # 1. Initialize Linear Layers
        if isinstance(module, nn.Linear):
            w_id = id(module.weight)
            if w_id not in seen_params:
                nn.init.xavier_uniform_(module.weight)
                # Apply depth-scaling (Small Init) to layers entering the residual stream
                if "out_proj" in name or "linear_2" in name:
                    with torch.no_grad():
                        module.weight.data.mul_(1.0 / math.sqrt(2 * n_layers))
                seen_params.add(w_id)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        # 2. Initialize Embedding Layers (if not already handled by weight-tying in Linear)
        elif isinstance(module, nn.Embedding):
            w_id = id(module.weight)
            if w_id not in seen_params:
                # Transformers ideally use normal dist for embeddings (Phase 20 Fix)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                seen_params.add(w_id)