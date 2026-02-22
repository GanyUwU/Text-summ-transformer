"""
Analyze why pretraining failed to improve model quality

The model trained for 10,000 steps but only learned for ~1,250 steps.
Loss plateaued and never recovered.
"""

import torch

print("="*70)
print("PRETRAINING FAILURE ANALYSIS")
print("="*70)

ckpt_best = torch.load('pretrain_weights_100k/pretrain_best.pt', map_location='cpu', weights_only=False)
ckpt_final = torch.load('pretrain_weights_100k/pretrain_final.pt', map_location='cpu', weights_only=False)

best_step = ckpt_best.get('step', 'N/A')
best_loss = ckpt_best.get('best_loss', 'N/A')
final_step = ckpt_final.get('step', 'N/A')
final_best_loss = ckpt_final.get('best_loss', 'N/A')

print(f"\nBest checkpoint:")
print(f"  Step: {best_step}")
print(f"  Loss: {best_loss:.4f}")

print(f"\nFinal checkpoint:")
print(f"  Step: {final_step}")
print(f"  Best loss ever: {final_best_loss:.4f}")
print(f"  Current loss:  (varies)")

# Analysis
total_steps = 10000
trained_steps = best_step
plateau_steps = total_steps - trained_steps

print(f"\n" + "-"*70)
print(f"ANALYSIS:")
print(f"-"*70)
print(f"Total steps: {total_steps}")
print(f"Actually training: {trained_steps} steps")
print(f"Loss plateaued/failed: {plateau_steps} steps ({100*plateau_steps/total_steps:.1f}%)")

print(f"""

ROOT CAUSES (pick one):

❌ Learning rate too LOW
   → Model learned slowly for 1250 steps then had nothing left to learn
   → Fix: Increase lr in pretrain_config.py (try 5e-4 or 1e-3)

❌ Batch size too LARGE
   → Model got stuck in local minimum
   → Fix: Decrease batch_size (try 2 or 1)

❌ Not enough diverse data
   → Model memorized first 1250 batches, then ran out of new patterns
   → Can't fix easily (would need bigger Wikipedia corpus)

⚠️ Weird loss dynamics
   → Model loss decreased but didn't generalize to actual text
   → This is a deeper architectural issue

RECOMMENDED FIX:

1. Check TensorBoard loss curve to see when it stopped improving:
   tensorboard --logdir=runs/pretrain_wiki
   
2. If loss dropped then plateaued: Increase LR
   
3. If loss never dropped much: Check tokenizer/data

4. Then retrain with config changes
""")
