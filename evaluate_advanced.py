import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# Local imports
from pretrain_config import get_finetune_config
from model import build_transformer
from tokenizer_utils import get_tokenizer
from checkpoint_utils import load_checkpoint
from train_summarization import SummarizationDataset
from inference import beam_search_decode, greedy_decode

from datasets import load_dataset
import nltk
from nltk.util import ngrams


def ensure_requirements():
    """Check for advanced metric dependencies and instruct user if missing."""
    missing = []
    
    try:
        import evaluate
    except ImportError:
        missing.append("evaluate")
        missing.append("rouge_score")
        
    try:
        import bert_score
    except ImportError:
        missing.append("bert_score")
        
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
        
    if missing:
        print("❌ Missing advanced evaluation dependencies.")
        print("Please run the following command to install them:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def generate_summaries(model, dataloader, tokenizer, device, num_samples, config, use_beam=True):
    """Generate model summaries for the given dataloader."""
    model.eval()
    results = []
    
    print(f"Generating summaries for {num_samples} samples...")
    # Progress bar
    pbar = tqdm(total=num_samples, desc="Generating")
    
    with torch.no_grad():
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
                
            enc_input_batch = batch['encoder_input'].to("cuda")
            enc_mask_batch = batch['encoder_mask'].to("cuda")
            label_batch = batch['label'] # cpu
            
            # Encode entire batch first (helps slightly)
            enc_output_batch = model.encode(enc_input_batch, enc_mask_batch)
            
            B = enc_input_batch.size(0)
            for b in range(B):
                if samples_processed >= num_samples:
                    break
                    
                enc_input = enc_input_batch[b:b+1]
                enc_mask = enc_mask_batch[b:b+1]
                enc_output = enc_output_batch[b:b+1]
                
                # Ground truth
                label = label_batch[b].tolist()
                # Remove padding and special tokens for text
                label_text = tokenizer.decode([tok for tok in label if tok not in [tokenizer.pad_id, tokenizer.eos_id, tokenizer.bos_id]])
                
                # Source text
                src = enc_input[0].cpu().tolist()
                src_text = tokenizer.decode([tok for tok in src if tok not in [tokenizer.pad_id, tokenizer.eos_id, tokenizer.bos_id]])
                
                # Generate
                if use_beam:
                    output_tokens = beam_search_decode(
                        model, enc_output, enc_mask, enc_input, tokenizer, 
                        config['tgt_seq_len'], "cuda", 
                        beam_size=config.get('beam_size', 4),
                        length_penalty=config.get('length_penalty', 0.8),
                        no_repeat_ngram=config.get('no_repeat_ngram', 3)
                    )
                else:
                    output_tokens = greedy_decode(
                        model, enc_output, enc_mask, enc_input, tokenizer, 
                        config['tgt_seq_len'], device, 
                        no_repeat_ngram=config.get('no_repeat_ngram', 3)
                    )
                    
                pred_text = tokenizer.decode(output_tokens)
                
                results.append({
                    "source": src_text,
                    "reference": label_text,
                    "prediction": pred_text
                })
                samples_processed += 1
                pbar.update(1)
            
    pbar.close()
    return results

from nltk.util import ngrams


def compute_copy_rate(source, summary):
    src_tokens = source.lower().split()
    sum_tokens = summary.lower().split()

    if len(sum_tokens) == 0:
        return 0.0

    copied = sum(tok in src_tokens for tok in sum_tokens)
    return copied / len(sum_tokens)


def novel_ngram_ratio(source, summary, n=4):
    src_tokens = source.lower().split()
    sum_tokens = summary.lower().split()

    if len(sum_tokens) < n:
        return 0.0

    src_ngrams = set(ngrams(src_tokens, n))
    sum_ngrams = list(ngrams(sum_tokens, n))

    novel = [g for g in sum_ngrams if g not in src_ngrams]

    return len(novel) / len(sum_ngrams)



def run_evaluation(results, device_str):
    """Compute ROUGE, BERTScore, and NLI-based Faithfulness."""
    import evaluate
    from rouge_score import rouge_scorer
    from bert_score import score as b_score
    
    preds = [r["prediction"] for r in results]
    refs = [r["reference"] for r in results]
    srcs = [r["source"] for r in results]
       
    scores = {}
    
    print("\n--- Computing Extractiveness Diagnostics ---")

    copy_rates = []
    novel4_scores = []

    for src, pred in zip(srcs, preds):
        copy_rates.append(compute_copy_rate(src, pred))
        novel4_scores.append(novel_ngram_ratio(src, pred, 4))

    scores["CopyRate"] = sum(copy_rates) / len(copy_rates)
    scores["Novel4gram"] = sum(novel4_scores) / len(novel4_scores)

    print(f"Copy Rate: {scores['CopyRate']:.4f}")
    print(f"Novel 4-gram Ratio: {scores['Novel4gram']:.4f}")
    
    # 1. ROUGE (Lexical Overlap) with Recall/Precision/F1
    print("\n--- Computing ROUGE ---")
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True
    )
    rouge_acc = {
        "rouge1": {"p": [], "r": [], "f": []},
        "rouge2": {"p": [], "r": [], "f": []},
        "rougeL": {"p": [], "r": [], "f": []},
        "rougeLsum": {"p": [], "r": [], "f": []},
    }
    for ref, pred in zip(refs, preds):
        r = scorer.score(ref, pred)
        for k in rouge_acc.keys():
            rouge_acc[k]["p"].append(r[k].precision)
            rouge_acc[k]["r"].append(r[k].recall)
            rouge_acc[k]["f"].append(r[k].fmeasure)

    def _avg(xs):
        return float(sum(xs) / max(1, len(xs)))

    rouge_table = {
        "ROUGE-1": {
            "Recall": _avg(rouge_acc["rouge1"]["r"]),
            "Precision": _avg(rouge_acc["rouge1"]["p"]),
            "F1-Score": _avg(rouge_acc["rouge1"]["f"]),
        },
        "ROUGE-2": {
            "Recall": _avg(rouge_acc["rouge2"]["r"]),
            "Precision": _avg(rouge_acc["rouge2"]["p"]),
            "F1-Score": _avg(rouge_acc["rouge2"]["f"]),
        },
        "ROUGE-L": {
            "Recall": _avg(rouge_acc["rougeL"]["r"]),
            "Precision": _avg(rouge_acc["rougeL"]["p"]),
            "F1-Score": _avg(rouge_acc["rougeL"]["f"]),
        },
        "ROUGE-Lsum": {
            "Recall": _avg(rouge_acc["rougeLsum"]["r"]),
            "Precision": _avg(rouge_acc["rougeLsum"]["p"]),
            "F1-Score": _avg(rouge_acc["rougeLsum"]["f"]),
        },
    }

    # Keep backward compatibility with current report fields (F1 values)
    scores["ROUGE-1"] = rouge_table["ROUGE-1"]["F1-Score"]
    scores["ROUGE-2"] = rouge_table["ROUGE-2"]["F1-Score"]
    scores["ROUGE-L"] = rouge_table["ROUGE-L"]["F1-Score"]
    scores["ROUGE-Lsum"] = rouge_table["ROUGE-Lsum"]["F1-Score"]
    scores["ROUGE-1-P"] = rouge_table["ROUGE-1"]["Precision"]
    scores["ROUGE-1-R"] = rouge_table["ROUGE-1"]["Recall"]
    scores["ROUGE-2-P"] = rouge_table["ROUGE-2"]["Precision"]
    scores["ROUGE-2-R"] = rouge_table["ROUGE-2"]["Recall"]
    scores["ROUGE-L-P"] = rouge_table["ROUGE-L"]["Precision"]
    scores["ROUGE-L-R"] = rouge_table["ROUGE-L"]["Recall"]
    scores["ROUGE-Lsum-P"] = rouge_table["ROUGE-Lsum"]["Precision"]
    scores["ROUGE-Lsum-R"] = rouge_table["ROUGE-Lsum"]["Recall"]
    scores["ROUGE_TABLE"] = rouge_table

    print("ROUGE table (avg over samples):")
    print("Metric\tRecall\tPrecision\tF1-Score")
    print(f"ROUGE-1\t{rouge_table['ROUGE-1']['Recall']:.4f}\t{rouge_table['ROUGE-1']['Precision']:.4f}\t{rouge_table['ROUGE-1']['F1-Score']:.4f}")
    print(f"ROUGE-2\t{rouge_table['ROUGE-2']['Recall']:.4f}\t{rouge_table['ROUGE-2']['Precision']:.4f}\t{rouge_table['ROUGE-2']['F1-Score']:.4f}")
    print(f"ROUGE-L\t{rouge_table['ROUGE-L']['Recall']:.4f}\t{rouge_table['ROUGE-L']['Precision']:.4f}\t{rouge_table['ROUGE-L']['F1-Score']:.4f}")
    print(f"ROUGE-Lsum\t{rouge_table['ROUGE-Lsum']['Recall']:.4f}\t{rouge_table['ROUGE-Lsum']['Precision']:.4f}\t{rouge_table['ROUGE-Lsum']['F1-Score']:.4f}")
    
    # 2. BERTScore (Semantic Paraphrasing)
    print("\n--- Computing BERTScore ---")
    P, R, F1 = b_score(preds, refs, lang="en", verbose=False, device=device_str)
    scores["BERTScore-P"] = P.mean().item()
    scores["BERTScore-R"] = R.mean().item()
    scores["BERTScore-F1"] = F1.mean().item()
    print(f"BERTScore F1: {scores['BERTScore-F1']:.4f} (RoBERTa baseline)")
    
    # 3. SummaC
    print("\n--- Computing SummaC ---")
    try:
        from summac.model_summac import SummaCZS
        model_zs = SummaCZS(granularity="sentence", model_name="vitc", device=device_str)
        summac_res = model_zs.score(srcs, preds)
        # Handle dict output or list output
        if isinstance(summac_res, dict) and "scores" in summac_res:
            summac_scores_list = summac_res["scores"]
        else:
            summac_scores_list = summac_res
        scores["SummaC"] = float(sum(summac_scores_list) / max(1, len(summac_scores_list)))
        print(f"SummaC: {scores['SummaC']:.4f}")
    except ImportError:
        print("Warning: summac package not installed. Skipping SummaC.")
        scores["SummaC"] = "Not Installed"
    except Exception as e:
        print(f"Warning: Failed to run SummaC: {e}")
        scores["SummaC"] = "Error"

    # 4. BARTScore (Generation Likelihood)
    print("\n--- Computing BARTScore ---")
    try:
        from transformers import BartTokenizer, BartForConditionalGeneration
        import torch.nn.functional as F
        
        bart_name = 'facebook/bart-large-cnn'
        print(f"Loading {bart_name} for BARTScore...")
        b_tok = BartTokenizer.from_pretrained(bart_name)
        b_model = BartForConditionalGeneration.from_pretrained(bart_name).to(device_str)
        b_model.eval()
        
        bs_scores = []
        with torch.no_grad():
            from tqdm import tqdm
            for src, pred in tqdm(zip(srcs, preds), total=len(srcs), desc="BARTScore"):
                inputs = b_tok([src], max_length=1024, truncation=True, return_tensors='pt').to(device_str)
                labels = b_tok([pred], max_length=256, truncation=True, return_tensors='pt').to(device_str)
                
                outputs = b_model(**inputs, labels=labels.input_ids)
                bs_scores.append(-outputs.loss.item())
                
        scores["BARTScore"] = sum(bs_scores) / len(bs_scores)
        print(f"BARTScore (Log Likelihood): {scores['BARTScore']:.4f} (Higher is better/more fluent)")
    except Exception as e:
        print(f"Warning: Failed to run BARTScore: {e}")
        scores["BARTScore"] = "Error"
        
    return scores


def main():
    parser = argparse.ArgumentParser(description="Advanced Summarization Evaluation (BERTScore, SummaC, ROUGE)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--samples", type=int, default=11000, help="Number of samples to evaluate")
    parser.add_argument("--beam", action="store_true", help="Explicitly use beam search")
    parser.add_argument("--greedy", action="store_true", help="Explicitly use greedy search")
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    args = parser.parse_args()
    
    ensure_requirements()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = get_finetune_config()
    tokenizer = get_tokenizer(config['tokenizer_model'])
    
    # Load checkpoint FIRST to extract dynamic sequence lengths safely
    print(f"Loading checkpoint metadata from: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if 'src_pos.pe' in state_dict:
        config['src_seq_len'] = state_dict['src_pos.pe'].shape[1]
    if 'tgt_pos.pe' in state_dict:
        config['tgt_seq_len'] = state_dict['tgt_pos.pe'].shape[1]

    # Load Model
    print(f"Building architecture... [Detected sizes: src={config['src_seq_len']}, tgt={config['tgt_seq_len']}]")
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=config['src_seq_len'],
        tgt_seq_len=config['tgt_seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
        share_weights=config.get('share_weights', True),
        use_copy=config.get('use_copy', True),
    ).to(device)
    
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    
    # Load test dataset (CNN/DailyMail)
    print(f"Loading {args.samples} test samples from cnn_dailymail...")
    try:
        dataset_subset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{args.samples}]")
    except Exception as e:
        print(f"Warning: Falling back to unversioned cnn_dailymail due to: {e}")
        dataset_subset = load_dataset("cnn_dailymail", split=f"test[:{args.samples}]")
    
    eval_dataset = SummarizationDataset(
        dataset_subset, tokenizer, config['src_seq_len'], config['tgt_seq_len'],
        lead_mask_prob=0.0
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Determine decoding method: follow config if not overridden by CLI flags
    use_beam = config.get('beam_size', 1) > 1
    if args.greedy:
        use_beam = False
    if args.beam:
        use_beam = True
        
    print(f"Decoding strategy: {'Beam Search (size=' + str(config.get('beam_size', 4)) + ')' if use_beam else 'Greedy'}")
    
    # Generate Summaries
    results = generate_summaries(model, dataloader, tokenizer, device, args.samples, config, use_beam=use_beam)
    
    # Calculate Metrics
    scores = run_evaluation(results, str(device))
    
    # Save Report
    report = {
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "decoding": "beam_search" if use_beam else "greedy",
        "beam_size": config.get('beam_size', 1) if use_beam else 1,
        "scores": scores,
        "examples": results[:10]  # Save top 10 examples for manual review
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Evaluation complete. Full report saved to {args.output}")


if __name__ == "__main__":
    main()
