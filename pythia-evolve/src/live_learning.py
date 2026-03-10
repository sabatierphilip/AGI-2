"""Continuous learning components with EWC."""

from __future__ import annotations

import gc
import json
import logging
from datetime import datetime, timezone

import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup

from config import CONFIG

LOGGER = logging.getLogger(__name__)


class EWC:
    """Elastic Weight Consolidation to prevent catastrophic forgetting."""

    def __init__(self, model: nn.Module, data_pipeline, n_samples: int = 200):
        self.fisher = {}
        self.optimal_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._compute_fisher(model, data_pipeline, n_samples)

    def _compute_fisher(self, model: nn.Module, data_pipeline, n_samples: int):
        for n, p in model.named_parameters():
            self.fisher[n] = torch.zeros_like(p)
        model.train()
        for _ in range(n_samples):
            batch = data_pipeline.get_mixed_batch(CONFIG.batch_size)
            out = model(**batch)
            model.zero_grad(set_to_none=True)
            out.loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2 / n_samples

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC quadratic regularization term."""
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.optimal_params and p.shape == self.optimal_params[n].shape:
                loss = loss + (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return CONFIG.ewc_lambda * 0.5 * loss

    def update_after_expansion(self, model: nn.Module, data_pipeline, n_samples: int = 100):
        """Refresh fisher and optimal parameter snapshots."""
        self.optimal_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._compute_fisher(model, data_pipeline, n_samples)


class ContinuousTrainer:
    """Three-phase continuous training orchestrator."""

    def __init__(self, model, tokenizer, data_pipeline, ewc):
        self.model = model
        self.tokenizer = tokenizer
        self.data_pipeline = data_pipeline
        self.ewc = ewc
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = CONFIG.log_dir / "training_log.jsonl"

    def training_step(self, model: nn.Module, batch: dict, optimizer, scheduler, ewc: EWC, step: int) -> dict:
        """Perform one optimization step with EWC penalty."""
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        ewc_loss = ewc.penalty(model)
        total = out.loss + ewc_loss
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()
        metrics = {
            "loss": float(out.loss.item()),
            "ewc_loss": float(ewc_loss.item()),
            "total_loss": float(total.item()),
            "perplexity": float(torch.exp(out.loss.detach()).item()),
            "grad_norm": grad_norm,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        self.log_training_state(step, metrics, "generic")
        return metrics

    def phase1_warmup(self, model: nn.Module) -> nn.Module:
        """Warm up the base model on early curriculum data."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(CONFIG.warmup_steps * 0.1), CONFIG.warmup_steps)
        for step in range(CONFIG.warmup_steps):
            batch = self.data_pipeline.get_curriculum_batch(0, CONFIG.warmup_steps)
            self.training_step(model, batch, optimizer, scheduler, self.ewc, step)
        return model

    def phase2_interleaved(self, model: nn.Module, expansion_name: str) -> nn.Module:
        """Run interleaved stabilization after an expansion op."""
        self.ewc.update_after_expansion(model, self.data_pipeline, n_samples=20)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)
        for step in range(100):
            batch = self.data_pipeline.get_curriculum_batch(step + 60, 100)
            metrics = self.training_step(model, batch, optimizer, scheduler, self.ewc, step)
            self.log_training_state(step, metrics, "interleaved")
        return model

    def phase3_consolidation(self, model: nn.Module, self_improve_loop) -> nn.Module:
        """Run full consolidation with periodic self-improvement."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 20, CONFIG.max_train_steps)
        for step in range(CONFIG.max_train_steps):
            batch = self.data_pipeline.get_curriculum_batch(step, CONFIG.max_train_steps)
            metrics = self.training_step(model, batch, optimizer, scheduler, self.ewc, step)
            if step > 0 and step % 100 == 0:
                model = self_improve_loop.run(model, self.tokenizer, self_improve_loop.cot_orchestrator, self.ewc)
                self.ewc.update_after_expansion(model, self.data_pipeline, n_samples=10)
                ckpt = CONFIG.checkpoint_dir / f"consolidation_step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt, safe_serialization=True)
        final = CONFIG.checkpoint_dir / "pythia-42m-trained"
        final.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final, safe_serialization=True)
        return model

    def log_training_state(self, step: int, losses: dict, phase: str):
        """Append one JSON line with training metrics."""
        row = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "clm_loss": losses["loss"],
            "ewc_loss": losses["ewc_loss"],
            "total_loss": losses["total_loss"],
            "perplexity": losses["perplexity"],
            "learning_rate": losses["learning_rate"],
            "grad_norm": losses["grad_norm"],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
