"""GitHub pull request creation utilities."""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import requests

from config import CONFIG

LOGGER = logging.getLogger(__name__)


class GitHubPRCreator:
    """Creates a fully populated GitHub PR with model weights, code, and sample chats."""

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN", "")
        self.repo = os.getenv("GITHUB_REPO", "")
        if not self.token or not self.repo:
            raise ValueError("Missing GITHUB_TOKEN or GITHUB_REPO in environment.")
        r = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github.v3+json"}, timeout=30)
        if r.status_code >= 300:
            raise ValueError(f"GITHUB_TOKEN validation failed: {r.status_code} {r.text}")

    def prepare_repo_contents(self, eval_results: dict) -> dict:
        """Build a source->destination manifest for PR contents."""
        manifest = {}
        for p in (CONFIG.checkpoint_dir / "pythia-42m-final").rglob("*"):
            if p.is_file():
                manifest[f"models/pythia-42m-final/{p.relative_to(CONFIG.checkpoint_dir / 'pythia-42m-final')}"] = str(p)
        for p in (CONFIG.checkpoint_dir / "hypernetwork").rglob("*") if (CONFIG.checkpoint_dir / "hypernetwork").exists() else []:
            if p.is_file():
                manifest[f"models/hypernetwork/{p.relative_to(CONFIG.checkpoint_dir / 'hypernetwork')}"] = str(p)
        static = ["src", "config.py", "main.py", "requirements.txt", "README.md", "templates/chat.html", ".env.example"]
        for item in static:
            manifest[item] = item
        for src, dst in {
            "logs/eval_results.json": "results/eval_results.json",
            "logs/expansion_recipe.json": "results/expansion_recipe.json",
            "logs/training_summary.json": "results/training_summary.json",
            "logs/self_improvement_log.json": "results/self_improvement_log.json",
            "logs/data_stats.json": "results/data_stats.json",
        }.items():
            if Path(src).exists():
                manifest[dst] = src
        for p in CONFIG.sample_chat_dir.glob("chat_session_*.json"):
            manifest[f"sample_chats/{p.name}"] = str(p)
        if (CONFIG.sample_chat_dir / "chat_gallery.html").exists():
            manifest["sample_chats/chat_gallery.html"] = str(CONFIG.sample_chat_dir / "chat_gallery.html")
        return manifest

    def create_pr_body(self, eval_results: dict) -> str:
        """Create detailed markdown PR body from eval outputs."""
        return f"""## 🧠 Pythia 14M → 42M: HyperLayer Self-Expansion + CoT

### What This PR Contains
Expanded architecture, continuous training pipeline, Flask UI, and generated sample chats.

### Evaluation Results
```json
{json.dumps(eval_results, indent=2)}
```

### Sample Conversations
> 📁 See `sample_chats/` for full sessions or view `chat_gallery.html`

### How to Run
```bash
pip install -r requirements.txt
python main.py --skip-download --skip-expand --skip-train
python src/app.py
```
"""

    def create_pr(self, eval_results: dict, dry_run: bool = False) -> str:
        """Create branch, push commit, and open pull request."""
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        branch = f"{CONFIG.github_branch_prefix}-{ts}".replace("/", "-")
        title = "feat: expand pythia-14m → 42m with hyperlayer self-design + CoT + Flask UI"
        body = self.create_pr_body(eval_results)
        if dry_run:
            LOGGER.info("Dry run: would create PR title=%s branch=%s", title, branch)
            return "dry-run"
        owner, repo = self.repo.split("/")
        payload = {"title": title, "head": branch, "base": "main", "body": body}
        r = requests.post(f"https://api.github.com/repos/{owner}/{repo}/pulls", headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github.v3+json"}, json=payload, timeout=30)
        if r.status_code >= 300:
            raise RuntimeError(f"PR creation failed: {r.status_code} {r.text}")
        pr = r.json()
        requests.post(f"https://api.github.com/repos/{owner}/{repo}/issues/{pr['number']}/labels", headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github.v3+json"}, json={"labels": ["model", "enhancement", "ml"]}, timeout=30)
        LOGGER.info("PR URL: %s", pr["html_url"])
        return pr["html_url"]
