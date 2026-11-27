"""Local vision preflight; generate and send two synthetic images only."""
import argparse
import base64
import hashlib
import io
import json
import re
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image, ImageDraw

LABELS = ("red_circle", "blue_square")
PROMPT = ("Identify the single shape in each image in attachment order. "
          "Return first and second. Labels: red_circle or blue_square.")
FORMAT = {"type": "object", "required": ["first", "second"],
          "additionalProperties": False,
          "properties": {k: {"type": "string", "enum": list(LABELS)}
                         for k in ("first", "second")}}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(port, path, payload=None):
    opener = build_opener(ProxyHandler({}), NoRedirect())
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(f"http://127.0.0.1:{port}{path}", data=data,
                  headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=120 if data else 5) as res:
        return json.load(res)


def fixtures():
    outputs = {}
    for label in LABELS:
        im = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(im)
        bounds = (96, 96, 416, 416)
        if label == LABELS[0]:
            draw.ellipse(bounds, fill="#e00000")
        else:
            draw.rectangle(bounds, fill="#0000e0")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        outputs[label] = buf.getvalue()
    return outputs


def preflight(model, digest, port, transport=request):
    version = transport(port, "/api/version")["version"]
    installed = transport(port, "/api/tags")["models"]
    matches = [r for r in installed if r.get("name") == model and r.get("digest") == digest
               and not r.get("remote_host") and not r.get("remote_model")]
    if len(matches) != 1:
        raise ValueError("requested model and digest do not match one locally installed model")
    images = fixtures()
    report = {"runtime": version, "model": model, "model_digest": digest,
              "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
              "synthetic_png_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in images.items()},
              "orders": []}
    for order in (LABELS, LABELS[::-1]):
        payload = {"model": model, "stream": False, "format": FORMAT,
                   "options": {"temperature": 0, "num_predict": 96},
                   "messages": [{"role": "user", "content": PROMPT,
                                 "images": [base64.b64encode(images[k]).decode("ascii") for k in order]}]}
        start = time.monotonic()
        entry = {"attachment_order": order, "passed": False}
        try:
            receipt = transport(port, "/api/chat", payload)
            entry["response"] = json.loads(receipt["message"]["content"])
            entry["passed"] = (
                receipt.get("done") is True and receipt.get("done_reason") == "stop"
                and receipt.get("model") == model
                and entry["response"] == dict(zip(("first", "second"), order)))
        except (OSError, ValueError, KeyError) as exc:
            entry["error"] = str(exc)
        entry["elapsed_ms"] = round((time.monotonic() - start) * 1000)
        report["orders"].append(entry)
    report["passed"] = all(x["passed"] for x in report["orders"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="exact installed Ollama tag")
    p.add_argument("--digest", required=True, help="full digest from /api/tags")
    p.add_argument("--port", type=int, default=11434)
    args = p.parse_args()
    if not re.fullmatch("[0-9a-f]{64}", args.digest) or "cloud" in args.model.lower():
        p.error("supply full lower-case SHA256 for a local model")
    if not 1 <= args.port <= 65535:
        p.error("invalid loopback port")
    try:
        report = preflight(args.model, args.digest, args.port)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"Local preflight not run: {exc}\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
