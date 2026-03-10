"""Download and verify base model."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG

LOGGER = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads and verifies EleutherAI/pythia-14m from HuggingFace."""

    def download(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Download, persist, and verify model by generation test."""
        out_dir = CONFIG.checkpoint_dir / "pythia-14m"
        out_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(CONFIG.base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.base_model_id)
        tokenizer.save_pretrained(out_dir)
        model.save_pretrained(out_dir, safe_serialization=True)
        arch = self.print_architecture(model)
        card = {
            "param_count": sum(p.numel() for p in model.parameters()),
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "architecture_type": model.config.model_type,
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            **arch,
        }
        (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

        ids = tokenizer("The meaning of life is", return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=30)
        LOGGER.info("Verification generation: %s", tokenizer.decode(gen[0], skip_special_tokens=True))
        return model, tokenizer

    def print_architecture(self, model) -> dict:
        """Log and return key architecture stats."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result = {
            "total_parameters": f"{total / 1e6:.2f}M",
            "trainable_parameters": trainable,
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "vocab_size": model.config.vocab_size,
            "max_position_embeddings": model.config.max_position_embeddings,
        }
        LOGGER.info("Architecture: %s", result)
        return result
