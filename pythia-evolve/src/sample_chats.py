"""Sample chat generation and gallery rendering."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

import torch

from config import CONFIG
from src.cot import CoTOrchestrator

LOGGER = logging.getLogger(__name__)

SAMPLE_CHAT_SCENARIOS = [
    {"name": "mathematical_reasoning", "description": "Step-by-step math problem solving", "turns": ["Can you help me solve a math problem?", "If I have 3 boxes with 12 apples each, and I give away 15 apples total, how many do I have left?", "Now what if I also find 8 more apples?", "Can you explain why you solved it that way?", "What's the percentage of apples I gave away from the original amount?", "Thanks! Can you give me a harder problem involving multiplication and division?"]},
    {"name": "logical_reasoning", "description": "Logic puzzles and deductive reasoning", "turns": ["I want to test your reasoning. Ready?", "All cats are animals. Whiskers is a cat. What can we conclude?", "Good. Now: Some animals can fly. Cats are animals. Can cats fly?", "Interesting. Here's a harder one: A doctor's brother has no brothers. What is the doctor's brother's profession?", "Why is that puzzle tricky for most people?", "Give me another logic puzzle to solve."]},
    {"name": "scientific_explanation", "description": "Science questions with chain-of-thought answers", "turns": ["Can you explain things scientifically?", "Why does ice float on water instead of sinking?", "How does that relate to why ice forms on top of lakes in winter?", "What would happen to aquatic life if water behaved like most liquids?", "Can you think through that step by step?", "That's fascinating. What other unusual properties does water have?"]},
]


class SampleChatGenerator:
    """Generates and saves realistic multi-turn sample chat sessions."""

    def __init__(self, model, tokenizer, cot_orchestrator: CoTOrchestrator):
        self.model = model
        self.tokenizer = tokenizer
        self.cot_orchestrator = cot_orchestrator

    def generate_response(self, message: str, history: list[dict], use_cot: bool = True) -> dict:
        """Generate one assistant response given history and message."""
        t0 = time.time()
        keywords = ["solve", "calculate", "why", "how", "explain", "prove", "what is", "how many", "if", "can you", "step by step"]
        should_cot = use_cot and any(k in message.lower() for k in keywords)
        if should_cot:
            trace = self.cot_orchestrator.think(message)
            content = "\n".join([s["thought"] for s in trace["steps"]]) + f"\nAnswer: {trace['answer']}"
            steps = trace["total_steps"]
        else:
            dialog = "\n".join([f"{m['role'].title()}: {m['content']}" for m in history])
            prompt = f"{dialog}\nHuman: {message}\nAssistant:" if dialog else f"Human: {message}\nAssistant:"
            ids = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=150, temperature=0.8, do_sample=True)
            content = self.tokenizer.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
            steps = 0
        return {
            "role": "assistant",
            "content": content.strip(),
            "used_cot": should_cot,
            "cot_steps": steps,
            "token_count": len(self.tokenizer.encode(content)),
            "generation_time_ms": int((time.time() - t0) * 1000),
        }

    def generate_session(self, scenario: dict) -> dict:
        """Generate one multi-turn sample session for a scenario."""
        history = []
        turns = []
        for i, msg in enumerate(scenario["turns"], start=1):
            resp = self.generate_response(msg, history, use_cot=True)
            history.extend([{"role": "human", "content": msg}, {"role": "assistant", "content": resp["content"]}])
            turns.append({"turn_number": i, "human": msg, "assistant": resp["content"], "used_cot": resp["used_cot"], "cot_steps": resp["cot_steps"], "token_count": resp["token_count"], "generation_time_ms": resp["generation_time_ms"]})
            time.sleep(0.5)
        return {
            "session_id": str(uuid.uuid4()),
            "scenario_name": scenario["name"],
            "description": scenario["description"],
            "model_version": "pythia-42m-final",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turns": turns,
            "total_turns": len(turns),
            "total_tokens": sum(t["token_count"] for t in turns),
            "mean_response_time_ms": sum(t["generation_time_ms"] for t in turns) / max(1, len(turns)),
        }

    def generate_all_sessions(self) -> list[dict]:
        """Generate and save all scenario sessions."""
        CONFIG.sample_chat_dir.mkdir(parents=True, exist_ok=True)
        sessions = [self.generate_session(s) for s in SAMPLE_CHAT_SCENARIOS]
        for i, s in enumerate(sessions, start=1):
            (CONFIG.sample_chat_dir / f"chat_session_{i}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        return sessions

    def render_html_gallery(self, sessions: list[dict]) -> str:
        """Render a self-contained static HTML gallery from sessions."""
        body = ["<html><head><meta charset='utf-8'><title>Pythia Chat Gallery</title>", "<style>body{background:#1a1a2e;color:#e0e0e0;font-family:system-ui;} .h{text-align:right;background:#2d6a9f;padding:8px;border-radius:10px;margin:8px;} .a{text-align:left;background:#2a2a4a;padding:8px;border-radius:10px;margin:8px;}</style></head><body>", "<h1>Pythia 42M — Self-Expanded with HyperLayers + CoT</h1>"]
        for s in sessions:
            body.append(f"<section><h2>{s['scenario_name']}</h2><p>{s['model_version']} | {s['timestamp']} | tokens={s['total_tokens']}</p>")
            for t in s["turns"]:
                body.append(f"<div class='h'>{t['human']}</div><div class='a'>{t['assistant']}<details><summary>View reasoning</summary>CoT steps: {t['cot_steps']}</details></div>")
            body.append("</section>")
        body.append("</body></html>")
        html = "\n".join(body)
        (CONFIG.sample_chat_dir / "chat_gallery.html").write_text(html, encoding="utf-8")
        return html
