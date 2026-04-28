"""
External Metrics — closing the internal/external correctness gap
================================================================
TRACEUM's internal diagnostics tell you HOW the model behaves mechanistically.
These external metrics tell you WHETHER it is correct.

Without this layer, the tool can only say:
  "model behaves like X"
Not:
  "model is correct"

Three tiers implemented here:
  1. ROUGE-1/2/L (fast, no dependencies beyond rouge-score)
     — word overlap proxy for informativeness
     — necessary but not sufficient

  2. Semantic similarity via sentence-transformers or fallback cosine
     — catches paraphrastic correct outputs that ROUGE misses
     — optional (requires sentence-transformers)

  3. Factual consistency (NLI-based, optional)
     — checks whether generated summary entails the source document
     — catches hallucinations that are fluent and semantically similar
     — requires transformers + a NLI model

Literature:
  ROUGE:    Lin 2004
  BERTScore: Zhang et al. 2019 (not implemented but referenced)
  Factual:  Maynez et al. 2020 (faithfulness in abstractive summarization)
  NLI probe: Falke et al. 2019
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    from sentence_transformers import SentenceTransformer as _ST
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


# ---------------------------------------------------------------------------
# 1. ROUGE (Lin 2004)
# ---------------------------------------------------------------------------

def compute_rouge(
    prediction: str,
    reference:  str,
) -> dict:
    """
    ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    Returns zeros with an explanatory note if rouge-score not installed.
    """
    if not _HAS_ROUGE:
        return {
            "rouge1": None, "rouge2": None, "rougeL": None,
            "note": "pip install rouge-score to enable ROUGE.",
        }

    scorer = _rouge_scorer_mod.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    result = scorer.score(reference, prediction)

    r1 = result["rouge1"].fmeasure
    r2 = result["rouge2"].fmeasure
    rL = result["rougeL"].fmeasure

    # Interpretation thresholds (CNN/DM domain, extractive upper bounds ~0.40)
    def rouge_label(r: float) -> str:
        if r >= 0.35: return "GOOD"
        if r >= 0.20: return "FAIR"
        return "POOR"

    return {
        "rouge1":       round(r1, 4),
        "rouge2":       round(r2, 4),
        "rougeL":       round(rL, 4),
        "rouge1_label": rouge_label(r1),
        "rouge2_label": rouge_label(r2),
        "warning": (
            "ROUGE measures word overlap only. "
            "A model can copy the source verbatim and score high. "
            "Combine with factual consistency for meaningful evaluation."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Semantic similarity (optional)
# ---------------------------------------------------------------------------

_st_model = None   # lazy load

def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    global _st_model
    if _st_model is None and _HAS_ST:
        _st_model = _ST(model_name)
    return _st_model


def compute_semantic_similarity(
    prediction: str,
    reference:  str,
    source:     Optional[str] = None,
) -> dict:
    """
    Cosine similarity between sentence embeddings.
    Catches paraphrastic correct outputs that ROUGE misses.

    Also computes faithfulness: sim(prediction, source) — whether the
    generated text is grounded in the source document.

    Falls back to trivial bag-of-words overlap if sentence-transformers
    is not installed (clearly labelled as approximation).
    """
    model = _get_st_model()

    if model is None:
        # Bag-of-words fallback — clearly marked as approximate
        def bow_sim(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb: return 0.0
            return len(sa & sb) / len(sa | sb)

        result = {
            "pred_ref_similarity": round(bow_sim(prediction, reference), 4),
            "method": "bag-of-words (approximate)",
            "note": "pip install sentence-transformers for proper semantic similarity",
        }
        if source is not None:
            result["pred_src_faithfulness"] = round(bow_sim(prediction, source), 4)
        return result

    texts = [prediction, reference]
    if source:
        texts.append(source)

    embeddings = model.encode(texts, convert_to_numpy=True)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    result = {
        "pred_ref_similarity": round(cos(embeddings[0], embeddings[1]), 4),
        "method": "sentence-transformers/all-MiniLM-L6-v2",
    }
    if source:
        result["pred_src_faithfulness"] = round(
            cos(embeddings[0], embeddings[2]), 4
        )
        # Faithfulness check: if similarity to source < 0.3, model may be hallucinating
        if result["pred_src_faithfulness"] < 0.30:
            result["faithfulness_alert"] = (
                f"Low source faithfulness ({result['pred_src_faithfulness']:.2f}). "
                f"Generated text may not be grounded in the source document."
            )

    return result


# ---------------------------------------------------------------------------
# 3. Factual consistency (NLI-based, Falke et al. 2019)
# ---------------------------------------------------------------------------

def compute_factual_consistency(
    prediction: str,
    source:     str,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> dict:
    """
    NLI-based faithfulness check: does the summary ENTAIL the source?

    Uses a cross-encoder NLI model to check:
      premise   = source document (truncated to model max_length)
      hypothesis = generated summary

    Labels: ENTAILMENT (faithful), NEUTRAL, CONTRADICTION (hallucination)

    Falke et al. 2019 showed NLI-based consistency outperforms
    QA-based and n-gram methods for detecting factual errors.

    Requires: pip install transformers torch
    Returns gracefully with a note if not available.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return {
            "factual_consistency": None,
            "note": (
                "pip install transformers torch for NLI-based factual consistency. "
                "This is the most important external metric for summarization."
            ),
        }

    try:
        nli = hf_pipeline(
            "text-classification",
            model=model_name,
            device=-1,           # CPU — avoid GPU dependency in diagnostic tool
        )
        # Truncate source to 512 chars to fit NLI model limits
        src_trunc = source[:512]
        result    = nli(f"{src_trunc} [SEP] {prediction}", truncation=True)[0]
        label     = result["label"].upper()
        score     = round(float(result["score"]), 4)

        return {
            "factual_consistency": label,
            "confidence": score,
            "is_faithful": label == "ENTAILMENT",
            "alert": (
                f"Factual inconsistency detected (label={label}, conf={score:.2f}). "
                f"Summary may contradict or confabulate facts from the source."
                if label != "ENTAILMENT" else None
            ),
        }
    except Exception as e:
        return {"factual_consistency": None, "error": str(e)}


# ---------------------------------------------------------------------------
# 4. Composite external report
# ---------------------------------------------------------------------------

def compute_external_metrics(
    prediction: str,
    reference:  Optional[str] = None,
    source:     Optional[str] = None,
    run_factual: bool = False,
) -> dict:
    """
    Full external metrics suite. Call this from the UI to close the
    internal/external correctness gap.

    Returns a structured dict with:
      - rouge (if reference provided)
      - semantic_similarity (if reference provided)
      - faithfulness (if source provided)
      - factual_consistency (if source provided AND run_factual=True)
      - overall_verdict: "FAITHFUL" | "UNFAITHFUL" | "UNKNOWN"
    """
    report: dict = {"prediction_length": len(prediction.split())}
    alerts: list = []

    if reference:
        report["rouge"] = compute_rouge(prediction, reference)
        report["semantic"] = compute_semantic_similarity(prediction, reference, source)
    elif source:
        report["semantic"] = compute_semantic_similarity(prediction, prediction, source)

    if source and run_factual:
        fc = compute_factual_consistency(prediction, source)
        report["factual_consistency"] = fc
        if fc.get("alert"):
            alerts.append(fc["alert"])

    # Faithfulness alert from semantic similarity
    sem = report.get("semantic", {})
    faith = sem.get("pred_src_faithfulness")
    if faith is not None and faith < 0.30:
        alerts.append(sem.get("faithfulness_alert", "Low source faithfulness."))

    report["alerts"] = alerts
    report["overall_verdict"] = (
        "UNKNOWN" if not alerts and not report.get("factual_consistency")
        else "UNFAITHFUL" if alerts
        else "FAITHFUL"
    )
    report["note"] = (
        "External metrics connect internal diagnostics to correctness. "
        "TRACEUM's internal signals describe HOW the model behaves; "
        "these metrics describe WHETHER the output is correct."
    )

    return report
