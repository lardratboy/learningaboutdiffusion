"""
Diffusion, live: a local Stable Diffusion server that streams every step to the browser.

    uv run server.py                  # then open http://127.0.0.1:8765
    uv run server.py --model <hf-id>  # any SD 1.x / 2.x style checkpoint in diffusers format

Wire format (WebSocket, server → client): one binary frame per event
    [u32 little-endian: header length N][N bytes UTF-8 JSON][pad to 4 bytes][payload]
The JSON header carries {"type", ...meta, "tensors": [{name, shape, offset, length}],
"blobs": [{name, mime, offset, length}]}. Offsets are relative to the payload start, and each
tensor is raw little-endian float32, so the browser can wrap it in a Float32Array without copying.

Client → server: JSON text frames {"type": "generate", ...params} or {"type": "cancel"}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sd_probe import GenParams, LiveDiffusion

HERE = Path(__file__).parent


def pack(kind: str, meta: dict, tensors: dict[str, np.ndarray], blobs: dict[str, bytes]) -> bytes:
    def pad4(n: int) -> int:
        return (4 - n % 4) % 4

    parts: list[bytes] = []
    offset = 0
    t_index, b_index = [], []
    for name, arr in tensors.items():
        data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        t_index.append({"name": name, "shape": list(arr.shape), "offset": offset, "length": len(data)})
        parts.append(data)
        offset += len(data)
    for name, data in blobs.items():
        b_index.append({"name": name, "mime": "image/jpeg", "offset": offset, "length": len(data)})
        parts += [data, b"\0" * pad4(len(data))]
        offset += len(data) + pad4(len(data))
    header = json.dumps({"type": kind, **meta, "tensors": t_index, "blobs": b_index}).encode()
    header += b" " * pad4(4 + len(header))
    return struct.pack("<I", len(header)) + header + b"".join(parts)


def build_app(model: LiveDiffusion) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=model.load, daemon=True, name="model-load").start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(HERE / "web" / "index.html")

    @app.get("/diffusion-walk.html")
    async def walk():
        return FileResponse(HERE.parent / "diffusion-walk.html")

    @app.get("/info")
    async def info():
        return model.info()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        cancel = threading.Event()
        job: asyncio.Future | None = None

        def emit(kind: str, meta: dict, tensors: dict, blobs: dict) -> None:
            # called from the GPU worker thread
            frame = pack(kind, meta, tensors, blobs)
            loop.call_soon_threadsafe(outbox.put_nowait, frame)

        async def sender():
            while (frame := await outbox.get()) is not None:
                await sock.send_bytes(frame)

        async def announce_until_ready():
            last = None
            while True:
                info = model.info()
                if info["status"] != last:
                    await outbox.put(pack("hello", {"info": info}, {}, {}))
                    last = info["status"]
                if info["ready"]:
                    return
                await asyncio.sleep(0.4)

        def run(params: GenParams) -> None:
            try:
                model.generate(params, emit, cancel)
            except Exception as e:  # report to the page instead of dying silently
                import traceback

                traceback.print_exc()
                emit("error", {"message": f"{type(e).__name__}: {e}"}, {}, {})

        send_task = asyncio.create_task(sender())
        hello_task = asyncio.create_task(announce_until_ready())
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if msg.get("type") == "cancel":
                    cancel.set()
                elif msg.get("type") == "generate":
                    if not model.ready:
                        await outbox.put(pack("error", {"message": f"model not ready yet ({model.status})"}, {}, {}))
                        continue
                    if job and not job.done():
                        cancel.set()
                        await job
                    cancel = threading.Event()
                    params = GenParams.from_dict(msg)
                    print(f"[gen] {params}", flush=True)
                    job = loop.run_in_executor(None, run, params)
        except WebSocketDisconnect:
            pass
        finally:
            cancel.set()
            hello_task.cancel()
            if job:
                await asyncio.shield(job)
            await outbox.put(None)
            send_task.cancel()

    app.mount("/static", StaticFiles(directory=HERE / "web"), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--preview", default="madebyollin/taesd", help="tiny VAE used for per-step previews")
    ap.add_argument("--size", type=int, default=512, help="image side in pixels (multiple of 64)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    model = LiveDiffusion(a.model, a.preview, a.size)
    print(f"[server] http://{a.host}:{a.port}  ·  device {model.device}  ·  model {a.model}", flush=True)
    uvicorn.run(build_app(model), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
