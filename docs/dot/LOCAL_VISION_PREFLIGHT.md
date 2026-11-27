# 本地 Qwen 双图预检：2026-11-27

Refs #31, Draft #32. Windows + NVIDIA RTX 4090 Laptop GPU (16 GiB); portable Ollama 0.40.18, cloud disabled confirmed by runtime log, used only loopback, no service left running. Qwen3-VL model and vision projector SHA256 verified; tag `qwen3-vl:4b-instruct-q4_K_M`, manifest `84f7ed8042fa7cdbd902f7dcb7d0370e032095047cc3244179f263b77b5bc318`. Model runtime reported 37/37 layers offloaded to GPU.

| Request | Expected | Actual | Wall time |
|---|---|---|---:|
| A then B, cold start | red_circle, blue_square | red_circle, blue_square | 70.032 s |
| B then A, warm | blue_square, red_circle | blue_square, red_circle | 0.812 s |

Same prompt and schema, reversed actual PNG bytes only, no retries on either inference. Both responses HTTP 200 with completed structured JSON and correct order. Timing describes these two different run states, not a portrait-scoring speed estimate. Prepared PNG SHA256: red `9a5ec25880ebcb61b70aebbb103d8a8cf57c8c3d5f1ca3c5a0a4647a79844188`; blue `7bb9a3af3983fb0ac93e997129d898ab643371288a30a071f92337e81f16a8c9`. Prompt SHA256 `91b3c7a5f3eb6a787a9137f2697c3158aaeec275528720c7c534d931605d0958`.

Reproduce with a local-only runtime and an installed pinned model: `python scripts/smoke_ollama_vision_pair.py --model qwen3-vl:4b-instruct-q4_K_M --digest 84f7ed8042fa7cdbd902f7dcb7d0370e032095047cc3244179f263b77b5bc318 --port <local-port>`. This script has no photo file argument and accepts only loopback. Do not add synthetic results to portrait metrics.

**Scope:** this verifies a second model family can consume two images in the correct order. It does not show that Qwen can grade portraits, that Gemini agrees with it, or that guidance improves photos. No portraits or paid calls. Next: collect authorized matched pairs, run the full frozen prompt locally and on the approved Gemini route where sample permissions allow, then compare to human blind ratings.
