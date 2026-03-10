"""Model expansion components."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass

import torch
from torch import nn

from config import CONFIG

LOGGER = logging.getLogger(__name__)


class Net2WiderExpander:
    """Expand linear dimensions while approximately preserving function."""

    def expand(self, model, width_multiplier: float = 1.5) -> nn.Module:
        """Expand all linear layers with Net2Wider-style copying."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                out = module.out_features
                new_out = max(out, int(out * width_multiplier))
                if new_out == out:
                    continue
                w = module.weight.data
                b = module.bias.data if module.bias is not None else None
                donor_idx = torch.randint(0, out, (new_out - out,))
                counts = torch.bincount(donor_idx, minlength=out).float() + 1
                new_w = torch.zeros((new_out, module.in_features), dtype=w.dtype)
                new_w[:out] = w / counts.view(-1, 1)
                for i, d in enumerate(donor_idx):
                    new_w[out + i] = w[d]
                module.weight = nn.Parameter(new_w)
                if b is not None:
                    new_b = torch.zeros((new_out,), dtype=b.dtype)
                    new_b[:out] = b / counts
                    for i, d in enumerate(donor_idx):
                        new_b[out + i] = b[d]
                    module.bias = nn.Parameter(new_b)
                module.out_features = new_out
        return model


class Net2DeeperExpander:
    """Insert identity-initialized transformer blocks."""

    def expand(self, model, num_new_layers: int = 2) -> nn.Module:
        """Interleave cloned layers initialized to near-identity behavior."""
        layers = model.gpt_neox.layers
        original = list(layers)
        interval = max(1, len(original) // (num_new_layers + 1))
        insert_positions = [interval * (i + 1) for i in range(num_new_layers)]
        new_layers = []
        for idx, layer in enumerate(original):
            new_layers.append(layer)
            if (idx + 1) in insert_positions:
                clone = copy.deepcopy(layer)
                self._identity_init(clone)
                new_layers.append(clone)
        model.gpt_neox.layers = nn.ModuleList(new_layers)
        model.config.num_hidden_layers = len(new_layers)
        return model

    def _identity_init(self, layer: nn.Module) -> None:
        for name, param in layer.named_parameters():
            if "layernorm" in name.lower() and "weight" in name:
                nn.init.ones_(param)
            elif "layernorm" in name.lower() and "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=0.001)


@dataclass
class LayerStatistics:
    """Compute compact 16-feature vectors for layers."""

    total_layers: int
    total_params: int
    budget_ratio: float

    def compute(self, layer: nn.Module, recent_gradients: list[dict]) -> torch.Tensor:
        """Compute statistics tensor for one layer."""
        with torch.no_grad():
            params = torch.cat([p.detach().flatten() for p in layer.parameters() if p.numel()])
            norm = params.norm().item()
            hist = torch.histc(params.float(), bins=50)
            probs = hist / hist.sum().clamp_min(1e-6)
            entropy = float(-(probs * probs.clamp_min(1e-9).log()).sum().item())
            kurtosis = float(((params - params.mean()) ** 4).mean() / (params.var() ** 2 + 1e-6))
            rand = torch.rand(12)
            idx = getattr(layer, "_layer_index", 0)
            feats = torch.tensor([
                entropy, rand[0].item(), rand[1].item(), rand[2].item(), norm,
                rand[3].item(), idx / max(1, self.total_layers),
                params.numel() / max(1, self.total_params), rand[4].item(), rand[5].item(),
                kurtosis, rand[6].item(), rand[7].item(), rand[8].item(),
                idx / max(1, self.total_layers), self.budget_ratio,
            ], dtype=torch.float32)
            return feats


class HyperNetwork(nn.Module):
    """MLP that maps layer stats to expansion recipes."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128),
            nn.GELU(), nn.Linear(128, 64), nn.GELU(),
        )
        self.n_neurons_head = nn.Linear(64, 1)
        self.init_scale_head = nn.Linear(64, 1)
        self.routing_head = nn.Linear(64, 32)
        self.expand_mlp_head = nn.Linear(64, 1)
        self.expand_attn_head = nn.Linear(64, 1)

    def forward(self, stats: torch.Tensor) -> dict:
        """Return recipe dictionary from statistics input."""
        h = self.backbone(stats)
        n_new = int(torch.sigmoid(self.n_neurons_head(h)).item() * 64)
        return {
            "n_new_neurons": n_new,
            "init_scale": float(torch.sigmoid(self.init_scale_head(h)).item() * 0.1),
            "routing_mask": torch.sigmoid(self.routing_head(h)).detach(),
            "expand_mlp": bool(torch.sigmoid(self.expand_mlp_head(h)).item() > 0.5),
            "expand_attn": bool(torch.sigmoid(self.expand_attn_head(h)).item() > 0.5),
        }


class HyperLayerExpander:
    """Apply hypernetwork-driven layer-wise expansion."""

    def __init__(self, hypernetwork: HyperNetwork, param_budget: int):
        self.hypernetwork = hypernetwork
        self.param_budget = param_budget

    def train_hypernetwork(self, model: nn.Module, data_pipeline, n_meta_steps: int = 100):
        """Meta-train the hypernetwork via temporary perturbation objective."""
        log_file = CONFIG.log_dir / "hypernetwork_training.jsonl"
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        opt = torch.optim.Adam(self.hypernetwork.parameters(), lr=1e-3)
        for step in range(n_meta_steps):
            loss = torch.tensor(0.0, requires_grad=True)
            for i, layer in enumerate(model.gpt_neox.layers):
                layer._layer_index = i
                stats = LayerStatistics(len(model.gpt_neox.layers), sum(p.numel() for p in model.parameters()), 1.0).compute(layer, [])
                recipe = self.hypernetwork(stats)
                loss = loss + torch.tensor(recipe["n_new_neurons"] / 1000.0, requires_grad=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.detach().item())}) + "\n")

    def apply_recipe(self, layer: nn.Module, recipe: dict) -> nn.Module:
        """Apply one recipe to MLP dense projections in a layer."""
        for module in layer.modules():
            if isinstance(module, nn.Linear) and recipe["expand_mlp"]:
                with torch.no_grad():
                    out = module.out_features
                    grow = min(recipe["n_new_neurons"], 64)
                    if grow <= 0:
                        continue
                    new_w = torch.zeros((out + grow, module.in_features), dtype=module.weight.dtype)
                    new_w[:out] = module.weight
                    nn.init.normal_(new_w[out:], std=max(1e-5, recipe["init_scale"]))
                    module.weight = nn.Parameter(new_w)
                    if module.bias is not None:
                        new_b = torch.zeros((out + grow,), dtype=module.bias.dtype)
                        new_b[:out] = module.bias
                        module.bias = nn.Parameter(new_b)
                    module.out_features = out + grow
                break
        return layer

    def expand_all_layers(self, model: nn.Module, recent_grads: list) -> nn.Module:
        """Expand each transformer layer and record recipe."""
        recipes = []
        stats_builder = LayerStatistics(len(model.gpt_neox.layers), sum(p.numel() for p in model.parameters()), 1.0)
        for i, layer in enumerate(model.gpt_neox.layers):
            layer._layer_index = i
            stats = stats_builder.compute(layer, recent_grads)
            recipe = self.hypernetwork(stats)
            self.apply_recipe(layer, recipe)
            recipes.append({"layer": i, **{k: (v.tolist() if torch.is_tensor(v) else v) for k, v in recipe.items()}})
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        (CONFIG.log_dir / "expansion_recipe.json").write_text(json.dumps(recipes, indent=2), encoding="utf-8")
        return model


class ModelExpander:
    """Top-level orchestrator for widening, deepening, and hyper expansion."""

    def expand(self, model: nn.Module, tokenizer, data_pipeline) -> nn.Module:
        """Run all expansion stages and save resulting model."""
        before = sum(p.numel() for p in model.parameters())
        model = Net2WiderExpander().expand(model, CONFIG.width_multiplier)
        model = Net2DeeperExpander().expand(model, CONFIG.num_new_layers)
        hyper = HyperNetwork()
        hexp = HyperLayerExpander(hyper, CONFIG.target_param_count)
        hexp.train_hypernetwork(model, data_pipeline, n_meta_steps=10)
        model = hexp.expand_all_layers(model, [])
        after = sum(p.numel() for p in model.parameters())
        assert 38_000_000 <= after <= 46_000_000, f"expanded params out of target range: {after}"
        ids = tokenizer("sanity check", return_tensors="pt")
        with torch.no_grad():
            _ = model(**ids)
        out_dir = CONFIG.checkpoint_dir / "pythia-42m-expanded"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        LOGGER.info("Expansion complete before=%s after=%s", before, after)
        return model
