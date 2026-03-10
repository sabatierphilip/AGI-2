"""Flask app for Pythia chat UI."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from src.cot import CoTOrchestrator
from src.sample_chats import SampleChatGenerator

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class ChatApp:
    """Stateful chat wrapper around model + response generation."""

    def __init__(self, model_path: str = None):
        bundled_path = Path(__file__).resolve().parents[1] / "models" / "pythia-42m-final"
        path = Path(model_path) if model_path else (CONFIG.checkpoint_dir / "pythia-42m-final")
        if not path.exists() and bundled_path.exists():
            LOGGER.info("Using bundled 42M model from %s", bundled_path)
            path = bundled_path
        if not path.exists():
            LOGGER.warning("Final model not found; falling back to base model")
            self.model = AutoModelForCausalLM.from_pretrained(CONFIG.base_model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.base_model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.cot = CoTOrchestrator(self.model, self.tokenizer)
        self.generator = SampleChatGenerator(self.model, self.tokenizer, self.cot)
        self.conversation_history = {}

    def generate_response(self, message: str, session_id: str) -> dict:
        """Generate and append response for a session."""
        history = self.conversation_history.setdefault(session_id, [])
        resp = self.generator.generate_response(message, history, use_cot=True)
        history.extend([{"role": "human", "content": message}, {"role": "assistant", "content": resp["content"]}])
        return resp


server = Flask(__name__, template_folder=str(CONFIG.template_dir))
chat_app = ChatApp()


@server.get("/")
def index():
    return render_template("chat.html")


@server.post("/chat")
def chat():
    try:
        payload = request.get_json(force=True)
        message = payload.get("message", "")
        session_id = payload.get("session_id") or str(uuid.uuid4())
        resp = chat_app.generate_response(message, session_id)
        return jsonify({"response": resp["content"], "session_id": session_id, "used_cot": resp["used_cot"], "cot_steps": resp["cot_steps"], "token_count": resp["token_count"], "generation_time_ms": resp["generation_time_ms"]})
    except Exception as exc:
        LOGGER.exception("chat failed")
        return jsonify({"error": str(exc)}), 500


@server.get("/health")
def health():
    params = sum(p.numel() for p in chat_app.model.parameters())
    return jsonify({"status": "ok", "model": "pythia-42m-final", "params": params})


@server.get("/reset")
def reset():
    chat_app.conversation_history.clear()
    return jsonify({"status": "cleared"})


@server.get("/sample_chats")
def list_sample_chats():
    CONFIG.sample_chat_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in CONFIG.sample_chat_dir.glob("chat_session_*.json"))
    return jsonify(files)


@server.get("/sample_chats/<filename>")
def get_sample_chat(filename: str):
    path = CONFIG.sample_chat_dir / filename
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(path)


if __name__ == "__main__":
    server.run(debug=False, port=CONFIG.flask_port, threaded=True)
