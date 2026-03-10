"""Chain-of-thought orchestration."""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from datetime import datetime, timezone

import torch

from config import CONFIG

LOGGER = logging.getLogger(__name__)


class CoTOrchestrator:
    """External-memory chain-of-thought loop."""

    STEP_PROMPT_TEMPLATE = """Question: {question}\n\n{previous_steps}Step {n}: """
    ANSWER_PROMPT_TEMPLATE = """Question: {question}\n\n{all_steps}\nBased on my reasoning above, the answer is:"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def think(self, question: str, temperature: float = 0.7) -> dict:
        """Generate multi-step reasoning trace and final answer."""
        memory, steps = [], []
        repetition = False
        for n in range(1, CONFIG.cot_max_steps + 1):
            prev = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(memory))
            prompt = self.STEP_PROMPT_TEMPLATE.format(question=question, previous_steps=(prev + "\n" if prev else ""), n=n)
            ids = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=CONFIG.cot_max_tokens_per_step, temperature=temperature, do_sample=True)
            text = self.tokenizer.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            reason = "continue"
            if "Answer:" in text:
                reason = "answer_found"
            for prior in memory:
                if self.jaccard_similarity(prior, text) > 0.8:
                    repetition = True
                    reason = "repetition"
                    break
            memory.append(text)
            steps.append({"step_number": n, "prompt": prompt, "thought": text, "token_count": len(self.tokenizer.encode(text)), "stopped_because": reason})
            if reason != "continue":
                break
        all_steps = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(memory))
        final_prompt = self.ANSWER_PROMPT_TEMPLATE.format(question=question, all_steps=all_steps)
        ids = self.tokenizer(final_prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=80)
        raw = self.tokenizer.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
        answer = self.extract_answer(raw)
        return {
            "question": question,
            "steps": steps,
            "answer": answer,
            "total_steps": len(steps),
            "total_tokens": sum(s["token_count"] for s in steps),
            "repetition_detected": repetition,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Token-level Jaccard similarity between two strings."""
        a, b = set(text_a.lower().split()), set(text_b.lower().split())
        return len(a & b) / max(1, len(a | b))

    def extract_answer(self, text: str) -> str:
        """Extract answer segment from generated text."""
        for pat in ["Answer:", "Therefore", "=", "the answer is"]:
            idx = text.lower().find(pat.lower())
            if idx >= 0:
                return text[idx + len(pat):].strip().split("\n")[0]
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return sentences[-1] if sentences else text.strip()


class SelfConsistencyCoT:
    """Majority-vote wrapper over multiple CoT traces."""

    def __init__(self, orchestrator: CoTOrchestrator, k: int = 3):
        self.orchestrator = orchestrator
        self.k = k

    def vote(self, question: str) -> dict:
        """Run k traces and aggregate answer by normalized vote."""
        traces = [self.orchestrator.think(question, temperature=0.7) for _ in range(self.k)]
        answers = [t["answer"] for t in traces]
        norm = ["".join(ch for ch in a.lower() if ch not in string.punctuation).strip() for a in answers]
        counts = Counter(norm)
        best_norm, count = counts.most_common(1)[0]
        idx = norm.index(best_norm)
        return {
            "question": question,
            "final_answer": answers[idx],
            "confidence": count / self.k,
            "all_answers": answers,
            "all_traces": traces,
            "k": self.k,
        }
