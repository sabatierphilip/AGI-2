"""Evaluation utilities for perplexity and reasoning performance."""

from __future__ import annotations

import json
import logging
import string

import torch
from datasets import load_dataset

from config import CONFIG
from src.cot import CoTOrchestrator, SelfConsistencyCoT
from src.self_improve import SelfImprovementLoop

LOGGER = logging.getLogger(__name__)


class Evaluator:
    """Evaluates baseline and expanded models."""

    def evaluate_perplexity(self, model, tokenizer, n_examples: int = 500) -> float:
        """Compute validation perplexity on WikiText-2."""
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        losses = []
        for row in ds.select(range(min(n_examples, len(ds)))):
            ids = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=256)
            if ids["input_ids"].shape[-1] < 2:
                continue
            with torch.no_grad():
                out = model(**ids, labels=ids["input_ids"])
            losses.append(float(out.loss.item()))
        return float(torch.exp(torch.tensor(sum(losses) / max(1, len(losses)))).item())

    def evaluate_cot_coherence(self, model, tokenizer, n_questions: int = 10) -> float:
        """Score trace quality over subset of question bank."""
        orch = CoTOrchestrator(model, tokenizer)
        sil = SelfImprovementLoop()
        qs = sil.QUESTION_BANK[:n_questions]
        scores = [sil.score_trace(orch.think(q), model, tokenizer) for q in qs]
        return sum(scores) / max(1, len(scores))

    def evaluate_answer_accuracy(self, model, tokenizer) -> float:
        """Evaluate fuzzy numeric correctness on deterministic math questions."""
        answers = {
            "What is 2^10?": "1024", "How many seconds in a day?": "86400",
            "What is 144 divided by 12, then multiplied by 3?": "36", "What is 15% of 240?": "36",
            "Solve: 3x + 7 = 22": "5", "What comes next: 2, 6, 18, 54, ?": "162",
            "If a train travels 60 mph for 2.5 hours, how far does it go?": "150",
            "What is the LCM of 12 and 18?": "36", "What is 347 multiplied by 28?": "9716",
            "Convert 0.375 to a fraction in lowest terms.": "3/8",
        }
        voter = SelfConsistencyCoT(CoTOrchestrator(model, tokenizer), k=3)
        ok = 0
        for q, expected in answers.items():
            pred = voter.vote(q)["final_answer"]
            norm = "".join(ch for ch in pred.lower() if ch not in string.punctuation)
            exp = "".join(ch for ch in expected.lower() if ch not in string.punctuation)
            ok += int(exp in norm)
        return ok / len(answers)

    def compare_models(self, original_model, final_model, tokenizer) -> dict:
        """Compare full metric suite and save report."""
        p0 = self.evaluate_perplexity(original_model, tokenizer)
        c0 = self.evaluate_cot_coherence(original_model, tokenizer)
        a0 = self.evaluate_answer_accuracy(original_model, tokenizer)
        p1 = self.evaluate_perplexity(final_model, tokenizer)
        c1 = self.evaluate_cot_coherence(final_model, tokenizer)
        a1 = self.evaluate_answer_accuracy(final_model, tokenizer)
        n0 = sum(p.numel() for p in original_model.parameters())
        n1 = sum(p.numel() for p in final_model.parameters())
        result = {
            "pythia_14m_original": {"perplexity": p0, "cot_coherence": c0, "answer_accuracy": a0, "param_efficiency": p0 / (n0 / 1e6), "param_count": n0},
            "pythia_42m_final": {"perplexity": p1, "cot_coherence": c1, "answer_accuracy": a1, "param_efficiency": p1 / (n1 / 1e6), "param_count": n1},
            "improvement": {
                "perplexity_pct": ((p0 - p1) / max(1e-6, p0)) * 100,
                "cot_coherence_pct": ((c1 - c0) / max(1e-6, c0)) * 100,
                "answer_accuracy_pct": ((a1 - a0) / max(1e-6, a0)) * 100,
            },
        }
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        (CONFIG.log_dir / "eval_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
