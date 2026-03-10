"""Self-improvement loop through trace scoring and distillation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from config import CONFIG
from src.cot import CoTOrchestrator

LOGGER = logging.getLogger(__name__)


class SelfImprovementLoop:
    """Generate, score, and train on model-produced CoT traces."""

    QUESTION_BANK = [
        "What is 347 multiplied by 28?", "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "What is the sum of the first 10 prime numbers?", "Solve: 3x + 7 = 22", "What is 15% of 240?",
        "If all birds can fly, and penguins are birds, can penguins fly? Analyze this.",
        "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
        "If it rains, the ground gets wet. The ground is wet. Did it rain? Explain.", "Why does ice float on water?",
        "What happens to pressure when volume decreases in a gas?", "A farmer has 17 sheep. All but 9 die. How many are left?",
        "What comes next: 2, 6, 18, 54, ?", "If you have a 3L jug and a 5L jug, how do you measure exactly 4L?",
        "What is the area of a circle with radius 7?", "Convert 0.375 to a fraction in lowest terms.", "What is 2^10?",
        "If a rectangle has area 48 and width 6, what is the perimeter?", "What is the LCM of 12 and 18?",
        "How many seconds in a day?", "What is 144 divided by 12, then multiplied by 3?",
    ]

    def __init__(self):
        self.cot_orchestrator = None

    def score_trace(self, trace: dict, model, tokenizer) -> float:
        """Score trace quality in [0,1]."""
        steps = [s["thought"] for s in trace.get("steps", [])]
        if not steps:
            return 0.0
        coherence = torch.exp(torch.tensor(-sum(len(s) for s in steps) / (len(steps) * 500))).item()
        sims = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                a, b = set(steps[i].split()), set(steps[j].split())
                sims.append(len(a & b) / max(1, len(a | b)))
        efficiency = 1.0 - (sum(s for s in sims if s > 0.5) / max(1, len([s for s in sims if s > 0.5])))
        ans = trace.get("answer", "")
        completeness = 1.0 if ans and len(ans) > 1 else 0.0
        markers = ["because", "therefore", "since", "if", "then", "however", "means", "implies"]
        text = " ".join(steps).lower()
        depth = min(1.0, sum(text.count(m) for m in markers) / max(1, len(steps) * 2))
        return 0.3 * coherence + 0.2 * efficiency + 0.3 * completeness + 0.2 * depth

    def generate_training_pairs(self, model, tokenizer, cot_orchestrator: CoTOrchestrator) -> list[dict]:
        """Generate best traces for questions and keep strong samples."""
        self.cot_orchestrator = cot_orchestrator
        rows, out_log = [], CONFIG.log_dir / "self_improve_scores.jsonl"
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        for q in self.QUESTION_BANK[: CONFIG.self_improve_questions_per_iter]:
            traces = [cot_orchestrator.think(q) for _ in range(3)]
            scores = [self.score_trace(t, model, tokenizer) for t in traces]
            best = max(range(len(scores)), key=lambda i: scores[i])
            if scores[best] >= 0.4:
                text = "\n".join([s["thought"] for s in traces[best]["steps"]]) + f"\nAnswer: {traces[best]['answer']}"
                rows.append({"input": q, "output": text})
            with out_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"question": q, "scores": scores}) + "\n")
        return rows

    def finetune_on_traces(self, model: nn.Module, tokenizer, training_pairs: list[dict], ewc, iteration: int) -> nn.Module:
        """Fine-tune model on synthetic traces with EWC loss."""
        if not training_pairs:
            return model
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        steps = min(len(training_pairs) * 2, 100)
        for s in range(steps):
            pair = training_pairs[s % len(training_pairs)]
            prompt = f"Question: {pair['input']}\nAnswer:"
            full = f"{prompt}\n{pair['output']}"
            ids = tokenizer(full, return_tensors="pt")
            plen = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
            labels = ids["input_ids"].clone()
            labels[:, :plen] = -100
            out = model(**ids, labels=labels)
            loss = out.loss + ewc.penalty(model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        ckpt = CONFIG.checkpoint_dir / f"self_improve_iter_{iteration}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt, safe_serialization=True)
        return model

    def run(self, model: nn.Module, tokenizer, cot_orchestrator, ewc) -> nn.Module:
        """Run full self-improvement iterations."""
        log_path = CONFIG.log_dir / "self_improvement_log.json"
        entries = []
        for i in range(CONFIG.self_improve_iterations):
            pairs = self.generate_training_pairs(model, tokenizer, cot_orchestrator)
            before = 0.0
            model = self.finetune_on_traces(model, tokenizer, pairs, ewc, i)
            after = 0.0
            entries.append({"iteration": i, "perplexity_before": before, "perplexity_after": after, "n_training_pairs": len(pairs)})
        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return model
