"""Main entrypoint for the Pythia self-expanding pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from src.app import server
from src.cot import CoTOrchestrator
from src.data import HighQualityDataPipeline
from src.download import ModelDownloader
from src.evaluate import Evaluator
from src.expand import ModelExpander
from src.github_pr import GitHubPRCreator
from src.live_learning import ContinuousTrainer, EWC
from src.sample_chats import SampleChatGenerator
from src.self_improve import SelfImprovementLoop

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def run_download(results, args):
    downloader = ModelDownloader()
    model_dir = CONFIG.checkpoint_dir / "pythia-14m"
    if model_dir.exists():
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        model, tokenizer = downloader.download()
    return {"model": model, "tokenizer": tokenizer}


def run_data_pipeline(results, args):
    tokenizer = results["Download"]["tokenizer"]
    pipeline = HighQualityDataPipeline(tokenizer)
    pipeline.load_all_sources()
    stats = pipeline.log_data_stats()
    return {"pipeline": pipeline, "stats": stats}


def run_expansion(results, args):
    model = results["Download"]["model"]
    tokenizer = results["Download"]["tokenizer"]
    pipeline = results["Data Pipeline"]["pipeline"]
    out_dir = CONFIG.checkpoint_dir / "pythia-42m-expanded"
    if out_dir.exists():
        model = AutoModelForCausalLM.from_pretrained(out_dir)
    else:
        model = ModelExpander().expand(model, tokenizer, pipeline)
    return {"model": model}


def run_training(results, args):
    model = results.get("Expansion", {}).get("model", results["Download"]["model"])
    tokenizer = results["Download"]["tokenizer"]
    pipeline = results["Data Pipeline"]["pipeline"]
    ewc = EWC(model, pipeline, n_samples=5)
    sil = SelfImprovementLoop()
    cot = CoTOrchestrator(model, tokenizer)
    sil.cot_orchestrator = cot
    trainer = ContinuousTrainer(model, tokenizer, pipeline, ewc)
    model = trainer.phase1_warmup(model)
    model = trainer.phase2_interleaved(model, "expansion")
    model = trainer.phase3_consolidation(model, sil)
    return {"model": model}


def run_evaluation(results, args):
    original = results["Download"]["model"]
    final_model = results.get("Training", {}).get("model", results.get("Expansion", {}).get("model", original))
    tokenizer = results["Download"]["tokenizer"]
    eval_results = Evaluator().compare_models(original, final_model, tokenizer)
    return eval_results


def run_sample_chats(results, args):
    model = results.get("Training", {}).get("model", results.get("Expansion", {}).get("model", results["Download"]["model"]))
    tokenizer = results["Download"]["tokenizer"]
    generator = SampleChatGenerator(model, tokenizer, CoTOrchestrator(model, tokenizer))
    sessions = generator.generate_all_sessions()
    generator.render_html_gallery(sessions)
    return {"sessions": len(sessions)}


def run_pr(results, args):
    return {"url": GitHubPRCreator().create_pr(results.get("Evaluation", {}), dry_run=args.dry_run)}


def print_summary_table(results):
    print("\nPipeline summary")
    for key, value in results.items():
        print(f"- {key}: {list(value.keys()) if isinstance(value, dict) else type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Pythia Self-Expanding LLM Pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-expand", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-sample-chats", action="store_true")
    parser.add_argument("--skip-pr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Skip pipeline, just start Flask server")
    args = parser.parse_args()

    if args.serve:
        server.run(port=CONFIG.flask_port, threaded=True)
        return

    results = {}
    steps = [
        ("Download", not args.skip_download, run_download),
        ("Data Pipeline", True, run_data_pipeline),
        ("Expansion", not args.skip_expand, run_expansion),
        ("Training", not args.skip_train, run_training),
        ("Evaluation", not args.skip_eval, run_evaluation),
        ("Sample Chats", not args.skip_sample_chats, run_sample_chats),
        ("GitHub PR", not args.skip_pr, run_pr),
    ]
    for step_name, should_run, step_fn in steps:
        if not should_run:
            print(f"  SKIP  {step_name}")
            continue
        t0 = time.time()
        try:
            step_result = step_fn(results, args)
            results[step_name] = step_result
            print(f"  DONE  {step_name:20s}  {time.time()-t0:.1f}s")
        except Exception as exc:
            logging.exception("Step %s failed", step_name)
            print(f"  FAIL  {step_name:20s}  {time.time()-t0:.1f}s  ERROR: {exc}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
