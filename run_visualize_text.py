import torch
from pathlib import Path
from visualize_attention import extract_all_attention, generate_html
from config_summarization import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
import webbrowser

ARTICLE = r"""
Pokémon at 30: Fans explain what the series means to them
Have you caught 'em all?

It's 30 years since a little game called Pocket Monsters launched in Japan - marking the start of a phenomenon that would evolve into a behemoth.

An animated TV series, movies, a trading card game and the mega-hit mobile game Pokémon Go! have all helped to win fans across the globe.

Reportedly the highest-grossing media franchise in history, Pokémon is still a cultural phenomenon today, reaching new generations of fans across the world.

BBC Newsbeat has been asking some of them why they love the series so much, why it appeals to so many people, and why it continues to prove so popular.
Pokémon has always been about playing the part of a trainer, catching and collecting monsters before battling them against others.

When the first games were released on Nintendo's Game Boy handheld in 1996, they weren't expected to be a huge hit.

But strong word-of-mouth and the console's low price helped it to sell more than one million copies in its first year on sale.

A popular animated series and the spin-off Trading Card Game (TCG) helped to turn it into a full-on craze so huge the press gave it a name - "Pokémania".

It became such a sensation that schools started to ban children from bringing the cards to the playground.

The brand sparked a second global trend with the launch of mobile phone game Pokémon GO, which used a device's GPS and camera to place monsters in the real world, in 2016.

That app has since been downloaded more than a billion times.

When the Covid-19 pandemic hit, there was an explosion in Pokémon-related content, and Pokémon TCG in particular saw a big increase in interest.
"""


def pad_and_tensorize(ids, seq_len, tokenizer):
    # handle tokenizers.Encoding objects
    if hasattr(ids, 'ids'):
        ids_list = ids.ids
    else:
        ids_list = list(ids)
    ids_list = ids_list[:seq_len-2]
    arr = [tokenizer.token_to_id('[CLS]')] if hasattr(tokenizer, 'token_to_id') else [tokenizer.token_to_id('[CLS]')]
    # fall back to adding BOS/EOS as ids if tokenizer doesn't have token_to_id
    try:
        bos = tokenizer.token_to_id('[BOS]')
    except Exception:
        bos = tokenizer.token_to_id('<s>') if hasattr(tokenizer, 'token_to_id') else 1
    try:
        eos = tokenizer.token_to_id('[EOS]')
    except Exception:
        eos = tokenizer.token_to_id('</s>') if hasattr(tokenizer, 'token_to_id') else 2

    seq = [bos] + ids_list + [eos]
    pad_len = max(0, seq_len - len(seq))
    seq = seq + [tokenizer.token_to_id('[PAD]') if hasattr(tokenizer, 'token_to_id') else 0] * pad_len
    return torch.tensor(seq, dtype=torch.long)


def main():
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tk_path = Path(cfg['tokenizer_file'].format('shared'))
    if tk_path.exists():
        tokenizer = Tokenizer.from_file(str(tk_path))
    else:
        # fallback: try shared tokenizer file name directly
        tokenizer = Tokenizer.from_file('tokenizer_summarization_shared.json')

    # Build model
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=cfg['src_seq_len'],
        tgt_seq_len=cfg['tgt_seq_len'],
        d_model=cfg['d_model'],
        N=cfg['num_layers'],
        h=cfg['num_heads'],
        dropout=cfg['dropout'],
        d_ff=cfg['d_ff']
    ).to(device)

    weights = latest_weights_file_path(cfg)
    if weights:
        ck = torch.load(weights, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print('Loaded weights:', weights)
    else:
        print('No weights found; using random init')

    # Tokenize article
    enc_ids = tokenizer.encode(ARTICLE)
    dec_ids = [tokenizer.token_to_id('[BOS]')] if hasattr(tokenizer, 'token_to_id') else [tokenizer.token_to_id('<s>')]

    enc_tensor = pad_and_tensorize(enc_ids, cfg['src_seq_len'], tokenizer).unsqueeze(0)
    dec_tensor = pad_and_tensorize([], cfg['tgt_seq_len'], tokenizer).unsqueeze(0)

    # Masks
    enc_mask = (enc_tensor != tokenizer.token_to_id('[PAD]') if hasattr(tokenizer, 'token_to_id') else enc_tensor != 0).unsqueeze(1).unsqueeze(1)
    dec_mask = torch.tril(torch.ones((1, cfg['tgt_seq_len'], cfg['tgt_seq_len']), dtype=torch.bool))

    attn = extract_all_attention(model, enc_tensor, enc_mask, dec_tensor, dec_mask, device)

    # Map ids back to tokens for display
    src_tokens = [tokenizer.decode([i]) for i in enc_ids[:50]]
    tgt_tokens = [''] * min(30, cfg['tgt_seq_len'])

    out_path = generate_html(attn, src_tokens, tgt_tokens, Path('attention_visualizer.html'))
    print('Saved visualization to', out_path)
    webbrowser.open(f'file://{Path(out_path).absolute()}')


if __name__ == '__main__':
    main()
