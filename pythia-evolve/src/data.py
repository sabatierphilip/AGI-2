"""High-quality streaming data pipeline."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Iterator

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG

LOGGER = logging.getLogger(__name__)


class HighQualityDataPipeline:
    """Streams and mixes quality-filtered training data from multiple datasets."""

    SOURCE_WEIGHTS = {
        "wikitext": 0.25,
        "arxiv": 0.25,
        "pg19": 0.20,
        "openwebmath": 0.30,
    }

    def __init__(self, tokenizer, seq_length: int = 512):
        """Initialize pipeline with tokenizer and sequence length."""
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.sources = {}
        self.iterators = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = AutoModelForCausalLM.from_pretrained(CONFIG.base_model_id).to(self.device)
        self.base_model.eval()

    def _extract_text(self, example: dict) -> str:
        for key in ("text", "content", "article", "body"):
            if key in example and isinstance(example[key], str):
                return example[key]
        return ""

    def load_all_sources(self) -> dict:
        """Load all configured sources as streaming iterables with preprocessing."""
        mapping = {
            "wikitext": ("wikitext", "wikitext-103-raw-v1", "train"),
            "arxiv": ("scientific_papers", "arxiv", "train"),
            "pg19": ("pg19", None, "train"),
            "openwebmath": ("openwebmath", None, "train"),
        }
        for name, (dataset_name, subset, split) in mapping.items():
            kwargs = {"path": dataset_name, "split": split, "streaming": True}
            if subset:
                kwargs["name"] = subset
            ds = load_dataset(**kwargs)
            self.sources[name] = ds
            self.iterators[name] = self._dataset_to_chunks(ds, name)
        return self.sources

    def _dataset_to_chunks(self, ds, source: str) -> Iterator[torch.Tensor]:
        for ex in ds:
            text = self._extract_text(ex)
            if not self.quality_filter({"text": text}, source):
                continue
            for chunk in self.tokenize_and_chunk(text):
                yield chunk

    def quality_filter(self, example: dict, source: str) -> bool:
        """Run quality gates tuned per source."""
        text = example.get("text", "")
        if len(text) < 200 or len(text) > 100_000:
            return False
        alpha_frac = sum(ch.isalpha() or ch.isspace() for ch in text) / max(1, len(text))
        if alpha_frac < 0.7:
            return False
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 20:
            return False
        for idx in range(len(words) - 3):
            if words[idx] == words[idx + 1] == words[idx + 2] == words[idx + 3]:
                return False
        if len(set(words)) / len(words) < 0.3:
            return False

        if source == "arxiv":
            anchors = ["theorem", "proof", "equation", "therefore", "we show", "lemma", "proposition"]
            if not any(a in text.lower() for a in anchors):
                return False
        if source == "openwebmath":
            if not any(m in text for m in ["$", "\\", "equation", "align"]):
                return False
            if self._estimate_perplexity(text) > 1000:
                return False
        if source == "pg19":
            sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]
            if len(sentences) < 10:
                return False
            lengths = [len(s.split()) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            if not (8 <= avg_len <= 40):
                return False
        return True

    def _estimate_perplexity(self, text: str) -> float:
        ids = self.tokenizer(text[:2048], return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.base_model(**ids, labels=ids["input_ids"])
        return float(torch.exp(out.loss).cpu().item())

    def tokenize_and_chunk(self, text: str) -> list[torch.Tensor]:
        """Tokenize text into overlapping chunks."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        stride = max(1, int(self.seq_length * 0.9))
        chunks = []
        for i in range(0, len(ids), stride):
            part = ids[i : i + self.seq_length]
            if len(part) < self.seq_length:
                break
            chunks.append(torch.tensor(part, dtype=torch.long))
        return chunks

    def _get_source_sample(self, source: str) -> torch.Tensor:
        if source not in self.iterators:
            self.load_all_sources()
        while True:
            try:
                return next(self.iterators[source])
            except StopIteration:
                self.iterators[source] = self._dataset_to_chunks(self.sources[source], source)

    def _build_batch(self, sources: list[str], batch_size: int) -> dict:
        rows = [self._get_source_sample(random.choice(sources)) for _ in range(batch_size)]
        input_ids = torch.stack(rows)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }

    def get_mixed_batch(self, batch_size: int) -> dict:
        """Sample sources by static mixing weights."""
        names = list(self.SOURCE_WEIGHTS)
        probs = [self.SOURCE_WEIGHTS[n] for n in names]
        sampled = random.choices(names, weights=probs, k=batch_size)
        return self._build_batch(sampled, batch_size)

    def get_curriculum_batch(self, step: int, total_steps: int) -> dict:
        """Sample a batch from easy-to-hard curriculum weights."""
        ratio = step / max(1, total_steps)
        if ratio < 0.2:
            weights = [0.5, 0.1, 0.2, 0.2]
        elif ratio < 0.6:
            weights = [0.25, 0.25, 0.20, 0.30]
        else:
            weights = [0.1, 0.35, 0.15, 0.40]
        names = ["wikitext", "arxiv", "pg19", "openwebmath"]
        sampled = random.choices(names, weights=weights, k=CONFIG.batch_size)
        return self._build_batch(sampled, CONFIG.batch_size)

    def log_data_stats(self) -> dict:
        """Compute and persist light-weight statistics for each source."""
        stats = {}
        for source in ["wikitext", "arxiv", "pg19", "openwebmath"]:
            lengths, pass_count, ppl = [], 0, []
            ds = self.sources.get(source)
            if ds is None:
                continue
            for idx, ex in enumerate(ds):
                if idx >= 1000:
                    break
                text = self._extract_text(ex)
                lengths.append(len(text))
                passed = self.quality_filter({"text": text}, source)
                pass_count += int(passed)
                if passed and len(ppl) < 20:
                    ppl.append(self._estimate_perplexity(text[:1024]))
            stats[source] = {
                "avg_text_length": sum(lengths) / max(1, len(lengths)),
                "vocab_diversity": 0.0,
                "avg_perplexity": sum(ppl) / max(1, len(ppl)),
                "quality_pass_rate": pass_count / max(1, len(lengths)),
            }
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        out = CONFIG.log_dir / "data_stats.json"
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        LOGGER.info("Wrote data stats to %s", out)
        return stats
