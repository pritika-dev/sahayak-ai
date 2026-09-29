"""
Posts sample finished-call webhooks (Vapi, Retell, Bland, ElevenLabs and
generic formats) to a running Sahayk server, exactly as a 14566 voice/IVR
platform would when a call ends.

    python tools/demo_calls.py                       # local server
    python tools/demo_calls.py https://your-app.onrender.com

Uses SAHAYK_WEBHOOK_KEY from the environment, or the demo key in config.py.
All sample calls are synthetic.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import config  # noqa: E402
from calls import sample_payloads  # noqa: E402

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
key = os.environ.get("SAHAYK_WEBHOOK_KEY", config.WEBHOOK_KEY)

for provider, payload in sample_payloads():
    req = urllib.request.Request(
        f"{base}/webhook/calls?provider={provider}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"{provider:<12} -> {resp.read().decode()}")
