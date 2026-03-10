"""Global configuration for the Pythia self-expanding pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application configuration values."""

    base_model_id: str = "EleutherAI/pythia-14m"
    target_param_count: int = 42_000_000
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    sample_chat_dir: Path = Path("./sample_chats")
    template_dir: Path = Path("./templates")
    width_multiplier: float = 1.5
    num_new_layers: int = 2
    batch_size: int = 4
    max_train_steps: int = 500
    learning_rate: float = 1e-4
    ewc_lambda: float = 0.1
    warmup_steps: int = 50
    data_sources: list = field(default_factory=lambda: [
        "wikitext-103-raw-v1",
        "scientific_papers/arxiv",
        "pg19",
        "openwebmath",
    ])
    max_tokens_per_source: int = 5_000_000
    cot_max_steps: int = 6
    cot_max_tokens_per_step: int = 80
    cot_temperature: float = 0.7
    cot_self_consistency_k: int = 3
    self_improve_iterations: int = 3
    self_improve_questions_per_iter: int = 20
    flask_port: int = 5000
    github_branch_prefix: str = "model/pythia-42m-expanded"
    num_sample_chats: int = 3
    sample_chat_turns: int = 6


CONFIG = Config()
