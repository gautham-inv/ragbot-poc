#!/bin/sh
set -eu

# Start the API, wait for it to come up, then call /warmup once.
# This runs at *container start* (not Docker build time), so it can reach Qdrant.

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

uvicorn backend.app:app --host "$HOST" --port "$PORT" &
PID="$!"

python3 - <<PY
import json
import time
import urllib.request
import urllib.error

port = ${PORT}
base = f"http://127.0.0.1:{port}"

def get(path: str, timeout: float = 2.0):
    req = urllib.request.Request(base + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body

# Wait for /health
deadline = time.time() + 40.0
ok = False
while time.time() < deadline:
    try:
        status, _ = get("/health")
        if status == 200:
            ok = True
            break
    except Exception:
        pass
    time.sleep(0.4)

if not ok:
    print("[entrypoint] /health not ready; skipping warmup")
else:
    # Cold start can take a while (HF model download / load). Retry with a long timeout.
    attempts = 5
    last_err = None
    for i in range(attempts):
        try:
            status, body = get("/warmup", timeout=120.0)
            msg = ""
            try:
                data = json.loads(body)
                msg = json.dumps(
                    {k: data.get(k) for k in ("bm25_loaded", "embedder_loaded", "qdrant_client_ready", "langfuse")},
                    ensure_ascii=True,
                )
            except Exception:
                msg = body[:200].replace("\\n", " ")
            print(f"[entrypoint] warmup status={status} {msg}")
            last_err = None
            break
        except Exception as e:
            last_err = e
            # Backoff: 1s, 2s, 4s, 8s...
            time.sleep(min(16.0, 2 ** i))
    if last_err is not None:
        print(f"[entrypoint] warmup failed after {attempts} attempts: {type(last_err).__name__}: {last_err}")
PY

wait "$PID"
