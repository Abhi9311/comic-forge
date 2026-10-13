---
title: Comic Forge
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---
# Comic Forge: AI Storyboard & Comic Generator

Story in -> LLM plans a character sheet + panels -> FLUX generates every panel -> download as PDF.

**Stack:** FastAPI (Python) Â· React + Vite Â· Hugging Face Inference API (Qwen2.5-72B for planning, FLUX.1-schnell for images) Â· Pillow (PDF layout)

## Run locally
Get a free token at https://huggingface.co/settings/tokens, then:
```bash
# terminal 1 - backend
cd backend && pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # PowerShell: $env:HF_TOKEN="hf_xxx"
uvicorn main:app --reload

# terminal 2 - frontend
cd frontend && npm install && npm run dev      # open http://localhost:5173
```

## Deploy (one server serves both)
```bash
cd frontend && npm install && npm run build      # creates frontend/dist
cd ../backend && uvicorn main:app --host 0.0.0.0 --port 8000
```
On Render: Web Service, build `pip install -r backend/requirements.txt && cd frontend && npm install && npm run build`, start `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`, env var `HF_TOKEN`.

## How character consistency works (demo talking point)
1. The LLM writes one fixed visual description per character (hair, clothes, colours).
2. That exact description is injected into the prompt of every panel the character appears in.
3. All panels share one seed, so composition/colour tendencies stay stable.
This gives *good but not perfect* consistency. Limits and upgrades: IP-Adapter / reference-image conditioning, PuLID, or a per-character LoRA.

## Known limits / next steps
Jobs are held in memory (use a DB like SQLite/Postgres + a queue for production), no auth, no speech bubbles yet.
