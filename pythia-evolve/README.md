# Pythia Evolve

A production-structured pipeline for expanding **EleutherAI/pythia-14m** toward ~42M parameters with:

- Net2Wider + Net2Deeper + HyperLayer expansion
- Streaming high-quality multi-source data pipeline
- Continuous training with EWC
- Chain-of-thought orchestration and self-consistency
- Self-improvement loop
- Flask chat UI
- Sample chat generation and static gallery
- GitHub PR automation

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py --skip-pr --dry-run
python src/app.py
```

Open http://localhost:5000.



## Bundled 42M Weights

This repository now includes a full checkpoint artifact at `models/pythia-42m-final/` containing `model.safetensors`, `config.json`, and tokenizer files, so the model can run without a separate download step.

## Sample Chats

This repository includes **real model-generated** sample sessions in `sample_chats/chat_session_1.json` through `chat_session_3.json` (generated locally with `distilgpt2`), plus a static viewer at `sample_chats/chat_gallery.html`.


Model verification metadata is written to `logs/model_verification.json` and embedded in each `sample_chats/chat_session_*.json`.
