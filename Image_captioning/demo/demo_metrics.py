"""Single-image caption metrics for the demo.

Metrics on a single image are only an educational approximation. For the report,
use the corpus-level eval.py script on the full TEST set.
"""
from __future__ import annotations

from typing import Dict, List


def _tok(s: str) -> List[str]:
    return [w.strip().lower() for w in s.replace(".", " ").replace(",", " ").split() if w.strip()]


def score_single_image(hypothesis: str, references: List[str]) -> Dict[str, float | str]:
    refs = [r.strip() for r in references if r.strip()]
    if not hypothesis.strip() or not refs:
        return {"note": "Cần caption dự đoán và ít nhất một caption tham chiếu."}

    out: Dict[str, float | str] = {}
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        refs_tok = [_tok(r) for r in refs]
        hyp_tok = _tok(hypothesis)
        smooth = SmoothingFunction().method3
        out["BLEU-1"] = float(sentence_bleu(refs_tok, hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smooth))
        out["BLEU-2"] = float(sentence_bleu(refs_tok, hyp_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        out["BLEU-3"] = float(sentence_bleu(refs_tok, hyp_tok, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth))
        out["BLEU-4"] = float(sentence_bleu(refs_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
    except Exception as e:
        out["BLEU"] = f"không tính được: {e}"

    try:
        from nltk.translate.meteor_score import meteor_score
        out["METEOR"] = float(meteor_score([_tok(r) for r in refs], _tok(hypothesis)))
    except Exception as e:
        out["METEOR"] = f"không tính được: {e}"

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        out["ROUGE-L"] = float(max(scorer.score(r, hypothesis)["rougeL"].fmeasure for r in refs))
    except Exception as e:
        out["ROUGE-L"] = f"không tính được: {e}"

    try:
        from pycocoevalcap.cider.cider import Cider
        cider = Cider()
        score, _ = cider.compute_score({0: refs}, {0: [hypothesis]})
        out["CIDEr"] = float(score)
    except Exception as e:
        out["CIDEr"] = f"không tính được: {e}"

    return out
