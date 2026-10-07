import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from motion_inbetweening.inference.pipeline import run_inference_pipeline
from motion_inbetweening.infra.loading.safetensors import load_safetensors, load_safetensors_from_hub
from motion_inbetweening.lightning_modules.seq2seq_module import Seq2SeqModule

load_dotenv()

DEFAULT_MODEL = "AnimajSAS/AIS_BI_LSTM_v0"
INBETWEEN_PATH = "/v1/inbetween"
HEALTH_PATH = "/health"


def main(model: str = DEFAULT_MODEL, host: str = "127.0.0.1", port: int = 8765, device: str = "cpu") -> None:
    """Load MODEL (a local safetensors directory or a HuggingFace repository ID) and serve it on HOST:PORT."""
    module = _load_module(model).to(device)
    handler = partial(InferenceRequestHandler, module, model, device, threading.Lock())
    server = ThreadingHTTPServer((host, port), handler)
    logger.info(f"Serving {model} on http://{host}:{port} (device: {device}). Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping the server.")
    finally:
        server.server_close()


class InferenceRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, module: Seq2SeqModule, model: str, device: str, lock: threading.Lock, *args, **kwargs):
        self.module = module
        self.model = model
        self.device = device
        self.lock = lock
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path != HEALTH_PATH:
            self._send_json(404, {"error": f"Unknown path: {self.path}"})
            return
        self._send_json(200, {"status": "ok", "model": self.model})

    def do_POST(self) -> None:
        if self.path != INBETWEEN_PATH:
            self._send_json(404, {"error": f"Unknown path: {self.path}"})
            return
        try:
            content_length = int(self.headers["Content-Length"])
            input_payload = json.loads(self.rfile.read(content_length))
        except (TypeError, ValueError) as e:
            self._send_json(400, {"error": f"The request body is not valid JSON: {e}"})
            return

        try:
            with self.lock:
                output_payload = run_inference_pipeline(input_payload, self.module, self.device)
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid inference request: {e}")
            self._send_json(400, {"error": str(e)})
            return
        except Exception as e:
            logger.exception("Inference failed")
            self._send_json(500, {"error": f"Inference failed: {e}"})
            return

        logger.info(f"Predicted {len(output_payload)} frames from {len(input_payload)} keyed frames.")
        self._send_json(200, {str(frame_id): frame for frame_id, frame in output_payload.items()})

    def log_message(self, format: str, *args) -> None:
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _load_module(model: str) -> Seq2SeqModule:
    model_path = Path(model)
    if model_path.exists():
        return load_safetensors(model_path)
    return load_safetensors_from_hub(model)


if __name__ == "__main__":
    typer.run(main)
