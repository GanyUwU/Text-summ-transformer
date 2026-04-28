"""
Failure Mode Classifier
=======================
Replaces the scalar "Risk Score" with a named failure taxonomy.

The core critique: a scalar score implies comparability, ordering, and a
decision boundary — none of which are justified for a mix of independent
signals. copy_rate=0.8 with entropy=0.2 produces score=moderate in any
weighted scheme, but the pattern is unambiguously "extractive collapse".

This module maps (copy_rate, entropy, repetition, coverage) to a named
failure mode with confidence level and per-signal evidence.

Taxonomy (mutually exclusive, ordered by severity):
  EXTRACTIVE_COLLAPSE  — high copy, low entropy, model is copying confidently
  HALLUCINATING        — low copy (generating), high entropy, uncertain
  REPETITION_LOOP      — high repetition regardless of copy/entropy
  ATTENTION_STARVED    — cross-attn uniform, model not reading source
  HEALTHY              — no dominant failure signal

Literature:
  See et al. 2017 — p_gen as extractive/abstractive boundary
  Malinin & Gales 2021 — predictive entropy decomposition
  Coverage loss taxonomy: Wu et al. 2016
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    name: str
    value: float
    label: str          # "HIGH" | "MED" | "LOW"
    is_anomalous: bool


@dataclass
class FailureDiagnosis:
    mode: str                          # primary failure mode name
    confidence: str                    # "HIGH" | "MEDIUM" | "LOW"
    signals: list[Signal]              # per-signal evidence
    description: str                   # one human sentence
    recommended_action: str            # concrete next step
    secondary_mode: Optional[str] = None


# ---------------------------------------------------------------------------
# Adaptive thresholds — fitted to a baseline distribution, not hardcoded
# ---------------------------------------------------------------------------

class AdaptiveThresholds:
    """
    Stores percentile-based thresholds learned from a calibration run.
    Falls back to conservative hardcoded values when no calibration exists.

    Usage:
        thresh = AdaptiveThresholds()
        thresh.fit(baseline_metrics_list)   # list of dicts from healthy runs
        classifier = FailureModeClassifier(thresh)
    """

    def __init__(self, baseline_path: str = "calibration_baseline.json"):
        # Defaults — conservative, architecture-agnostic
        self.copy_high   = 0.65   # above = high copying
        self.copy_low    = 0.20   # below = pure generation
        self.entropy_high = 0.75  # above = uncertain
        self.entropy_low  = 0.20  # below = overconfident
        self.rep_high    = 0.25   # above = repetitive
        self.cov_low     = 0.15   # below = attention reuse (bad)
        self.attn_H_low  = 0.10   # below = cross-attn collapsed (norm_ent)
        self.attn_H_high = 0.90   # above = cross-attn uniform
        self._fitted     = False
        
        # Auto-load if path exists
        import os
        import json
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict) and "records" in data:
                    self.fit(data["records"])
                    self.metadata = data.get("metadata", {})
                else:
                    # Legacy fallback
                    self.fit(data)
                    self.metadata = {}
                    
                print(f"✅ FailureClassifier: Loaded baseline from {baseline_path}")
                if self.metadata.get('training_phase'):
                    print(f"   Phase context: {self.metadata['training_phase'].upper()}")
            except Exception as e:
                print(f"⚠️ FailureClassifier: Failed to load baseline ({e}). Using defaults.")
                self.metadata = {}
        else:
            self.metadata = {}

    def fit(self, baseline_metrics: list[dict]) -> None:
        """
        Learn thresholds from a list of baseline metric dicts.
        Each dict: {copy_rate, norm_entropy, repetition, norm_coverage, cross_attn_norm_ent}
        """
        if len(baseline_metrics) < 5: # Lowered for early testing
            return
            
        def pct(key, p): 
            vals = [m[key] for m in baseline_metrics if key in m]
            if not vals: return getattr(self, key, 0.5) # Fallback to existing default
            return float(np.percentile(vals, p))

        self.copy_high    = pct("copy_rate",        95)
        self.copy_low     = pct("copy_rate",         5)
        self.entropy_high = pct("norm_entropy",      95)
        self.entropy_low  = pct("norm_entropy",       5)
        self.rep_high     = pct("repetition",        95)
        self.cov_low      = pct("norm_coverage",      5)
        self.attn_H_low   = pct("cross_attn_norm_ent", 5)
        self.attn_H_high  = pct("cross_attn_norm_ent", 95)
        self._fitted      = True

    @property
    def source(self) -> str:
        return "fitted" if self._fitted else "default"


# ---------------------------------------------------------------------------
# Failure mode rules
# ---------------------------------------------------------------------------

def _label(value: float, low: float, high: float) -> str:
    if value >= high:   return "HIGH"
    if value <= low:    return "LOW"
    return "MED"


class FailureModeClassifier:
    """
    Maps a set of signals to a named failure mode.
    Does NOT produce a scalar score. The primary output is a FailureDiagnosis.
    """

    SEVERITY_RANK = {
        "EXTRACTIVE_COLLAPSE": 1,
        "HALLUCINATING":       2,
        "REPETITION_LOOP":     3,
        "ATTENTION_STARVED":   4,
        "HEALTHY":             5,
    }

    def __init__(self, thresholds: Optional[AdaptiveThresholds] = None):
        import os
        # If no thresholds provided, try to find a local baseline file
        if thresholds is None:
            self.t = AdaptiveThresholds("calibration_baseline.json")
        else:
            self.t = thresholds

    @staticmethod
    def get_calibrated_classifier(path="calibration_baseline.json"):
        return FailureModeClassifier(AdaptiveThresholds(path))

    def classify(
        self,
        copy_rate:           float,            # mean(1 - p_gen), [0,1]
        norm_entropy:        float,            # H_output / log(vocab), [0,1]
        repetition:          float,            # 1 - unique_ngram_ratio, [0,1]
        norm_coverage:       float,            # coverage_penalty / max_possible, [0,1]
        cross_attn_norm_ent: Optional[float] = None,   # [0,1] per head
    ) -> FailureDiagnosis:

        t = self.t

        # Build signal evidence
        signals = [
            Signal("copy_rate",    copy_rate,    _label(copy_rate,    t.copy_low,    t.copy_high),    copy_rate > t.copy_high),
            Signal("entropy",      norm_entropy, _label(norm_entropy, t.entropy_low, t.entropy_high), False),  # context-dependent
            Signal("repetition",   repetition,   _label(repetition,   0.0,           t.rep_high),     repetition > t.rep_high),
            Signal("coverage",     norm_coverage, _label(norm_coverage, t.cov_low,   1.0),            norm_coverage < t.cov_low),
        ]
        if cross_attn_norm_ent is not None:
            signals.append(Signal(
                "cross_attn_entropy", cross_attn_norm_ent,
                _label(cross_attn_norm_ent, t.attn_H_low, t.attn_H_high),
                cross_attn_norm_ent < t.attn_H_low or cross_attn_norm_ent > t.attn_H_high
            ))

        # Rule 1: EXTRACTIVE_COLLAPSE
        # High copy, low entropy = model is copying confidently (the silent killer)
        if copy_rate > t.copy_high and norm_entropy < t.entropy_low:
            conf = "HIGH" if (copy_rate > t.copy_high * 1.1) else "MEDIUM"
            return FailureDiagnosis(
                mode="EXTRACTIVE_COLLAPSE",
                confidence=conf,
                signals=signals,
                description=(
                    f"Model is copying {copy_rate*100:.0f}% of output with low uncertainty "
                    f"(norm_entropy={norm_entropy:.2f}). This is extractive behaviour, "
                    f"not moderate risk."
                ),
                recommended_action=(
                    "Increase lambda_p (copy penalty) in training config. "
                    "Check p_gen trajectory — early tokens should generate, not copy. "
                    "Verify copy_warmup_steps is respected."
                ),
                secondary_mode="REPETITION_LOOP" if repetition > t.rep_high else None,
            )

        # Rule 2: REPETITION_LOOP
        # High repetition is primary regardless of copy/entropy
        if repetition > t.rep_high:
            conf = "HIGH" if repetition > t.rep_high * 1.5 else "MEDIUM"
            return FailureDiagnosis(
                mode="REPETITION_LOOP",
                confidence=conf,
                signals=signals,
                description=(
                    f"High n-gram repetition ({repetition*100:.0f}%) detected. "
                    f"Coverage penalty={norm_coverage:.2f} suggests "
                    f"{'source token reuse' if norm_coverage < t.cov_low else 'output loop'}."
                ),
                recommended_action=(
                    "Increase coverage_loss_weight in training. "
                    "Apply no_repeat_ngram_size >= 3 at inference. "
                    "Check if cross-attention is attending to the same source tokens repeatedly."
                ),
            )

        # Rule 3: ATTENTION_STARVED
        # Cross-attention is collapsed or uniform — model not reading source
        if cross_attn_norm_ent is not None:
            if cross_attn_norm_ent < t.attn_H_low:
                return FailureDiagnosis(
                    mode="ATTENTION_STARVED",
                    confidence="HIGH",
                    signals=signals,
                    description=(
                        f"Cross-attention entropy too low (norm={cross_attn_norm_ent:.2f}). "
                        f"Decoder is not reading the source — likely stuck on BOS token."
                    ),
                    recommended_action=(
                        "Check encoder output norm. Inspect cross-attn heatmap for BOS collapse. "
                        "Verify encoder is not frozen. Consider entropy regularization on cross-attn."
                    ),
                )
            if cross_attn_norm_ent > t.attn_H_high:
                return FailureDiagnosis(
                    mode="ATTENTION_STARVED",
                    confidence="MEDIUM",
                    signals=signals,
                    description=(
                        f"Cross-attention entropy too high (norm={cross_attn_norm_ent:.2f}). "
                        f"Decoder is attending uniformly — not using the source."
                    ),
                    recommended_action=(
                        "Check that encoder is producing meaningful representations. "
                        "Run logit lens to see if encoder output is informative. "
                        "Inspect W_OV Frobenius norm for dead encoder heads."
                    ),
                )

        # Rule 4: HALLUCINATING
        # Low copy (generating), high entropy
        if copy_rate < t.copy_low and norm_entropy > t.entropy_high:
            return FailureDiagnosis(
                mode="HALLUCINATING",
                confidence="MEDIUM",
                signals=signals,
                description=(
                    f"Model is generating (copy_rate={copy_rate:.2f}) but uncertain "
                    f"(norm_entropy={norm_entropy:.2f}). Risk of confabulation."
                ),
                recommended_action=(
                    "Check training data coverage for this input domain. "
                    "Consider temperature reduction at inference. "
                    "Verify label smoothing is not too high (currently reducing confidence)."
                ),
            )

        # Rule 5: HEALTHY
        return FailureDiagnosis(
            mode="HEALTHY",
            confidence="HIGH",
            signals=signals,
            description="No dominant failure pattern detected across signals.",
            recommended_action="Monitor coverage penalty over longer decode sequences.",
        )

    def severity_rank(self, mode: str) -> int:
        """Integer rank for sorting. 1=worst, 5=healthy. Use this, not a float score."""
        return self.SEVERITY_RANK.get(mode, 5)
