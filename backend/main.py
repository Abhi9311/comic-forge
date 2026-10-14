"""Comic Forge API: story -> LLM storyboard plan -> consistent panel images -> PDF."""
import base64, datetime, io, json, os, re, textwrap, threading, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# Load tokens from backend/.env (locally). On Render, use the Environment tab.
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")
_txt = Path(__file__).parent / "token.txt"
if _txt.exists() and _txt.read_text().strip().startswith("hf_"):
    os.environ["HF_TOKEN"] = _txt.read_text().strip()
_tok = os.getenv("HF_TOKEN", "")
if not _tok.startswith("hf_"):
    print("\n*** HF_TOKEN missing or invalid. Set it in .env / token.txt / Render Environment ***\n")
else:
    print("HF_TOKEN loaded OK", flush=True)

# CF_ACCOUNT_ID from Render > Environment (or .env) wins. The built-in ID is only a
# fallback when nothing is set. CF_API_TOKEN must belong to the same Cloudflare account.
os.environ.setdefault("CF_ACCOUNT_ID", "14b97fa06310191d95d348701aa513a9")
_cid = os.getenv("CF_ACCOUNT_ID", "").strip()
print(f"Cloudflare account id length={len(_cid)} ends={_cid[-4:]} token_set={bool(os.getenv('CF_API_TOKEN', '').strip())}", flush=True)

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
# Story-planning models, tried in order (older Llama 3.x models were deprecated on 2026-05-30)
CF_CHAT_MODELS = ["@cf/meta/llama-4-scout-17b-16e-instruct", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"]
STORAGE = Path(__file__).resolve().parent / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)
STYLES = {
    "Comic Book": "vibrant American comic book art, bold ink outlines, halftone shading, dynamic composition",
    "Manga": "black and white manga, screentone shading, expressive linework",
    "Storybook": "soft watercolor children's storybook illustration, warm palette",
    "Noir": "film noir graphic novel, high contrast black and white ink, dramatic shadows",
    "3D Animation": "3D animated movie still, soft lighting, expressive characters, pastel colors",
    "Retro Sci-Fi": "1960s retro sci-fi pulp cover art, painted, bold colors",
}
JOBS: dict[str, dict] = {}
BODIES: dict[str, "StoryIn"] = {}   # original request per job (for retry)
PLANS: dict[str, dict] = {}         # storyboard per job (so retry doesn't re-plan)

# A provider is PAUSED for a while after an error, never switched off forever.
_OFF = {"cf": 0.0, "hf": 0.0, "poll": 0.0}
_LAST = {}  # last error per provider (shown on /api/health)

QUOTA_MSG = "Daily free image quota reached. Please try again after 5:30 AM IST."
BUSY_MSG = "Image services are temporarily unavailable. Please try again in a few minutes."


def _is_on(name: str) -> bool:
    return time.time() >= _OFF[name]


def _next_utc_midnight() -> float:
    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.timestamp()


def _pause(name: str, err) -> None:
    # FIX 2: smarter pauses. Daily quota -> until reset. Pollinations 402 -> 1 hour.
    s = str(err)
    _LAST[name] = s[:200]
    if name == "cf" and ("daily free allocation" in s or "neurons" in s):
        _OFF[name] = _next_utc_midnight()
        return
    if name == "poll" and "402" in s:
        _OFF[name] = time.time() + 3600
        return
    if any(c in s for c in ("401", "403")):
        secs = 300
    elif "402" in s:
        secs = 600
    elif "429" in s:
        secs = 20 if name == "cf" else 90
    else:
        secs = 0
    if secs:
        _OFF[name] = time.time() + secs


def cf_ready() -> bool:
    return bool(os.getenv("CF_ACCOUNT_ID", "").strip() and os.getenv("CF_API_TOKEN", "").strip())


def cloudflare_image(prompt: str, seed: int) -> Image.Image:
    """Cloudflare Workers AI (FLUX.1 schnell), free daily quota."""
    acct = os.getenv("CF_ACCOUNT_ID", "").strip()
    token = os.getenv("CF_API_TOKEN", "").strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{CF_IMAGE_MODEL}"
    body = json.dumps({"prompt": prompt[:2000], "steps": 4}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "comic-forge",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        raise RuntimeError(f"HTTP {e.code}: {detail}")
    try:  # FLUX returns JSON with a base64 image
        data = json.loads(raw)
        if not data.get("success", True):
            raise RuntimeError(f"Cloudflare error: {str(data.get('errors'))[:200]}")
        raw = base64.b64decode(data["result"]["image"])
    except (ValueError, KeyError, TypeError):
        pass  # some models return the image bytes directly
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pollinations_image(prompt: str, seed: int) -> Image.Image:
    """Free fallback (anonymous tier, may now require payment/key)."""
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:900])
           + f"?width=768&height=768&seed={seed}&model=flux&nologo=true")
    last = None
    for _ in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "comic-forge"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 429:   # 402 etc: don't waste time retrying
                break
            time.sleep(16)
        except Exception as e:
            last = e
            time.sleep(5)
    raise RuntimeError(f"Free image service failed: {last}")


_PIPE = {"p": None}
_LOCK = threading.Lock()


def local_image(prompt: str, seed: int) -> Image.Image:
    """Last resort: SD-Turbo on your own PC. Not available on Render (no torch)."""
    import torch
    from diffusers import AutoPipelineForText2Image
    with _LOCK:
        if _PIPE["p"] is None:
            gpu = torch.cuda.is_available()
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo", variant="fp16",
                torch_dtype=torch.float16 if gpu else torch.float32)
            _PIPE["p"] = pipe.to("cuda" if gpu else "cpu")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return _PIPE["p"](prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
                          width=512, height=512, generator=gen).images[0]


def make_image(client, prompt: str, seed: int, short: str) -> Image.Image:
    errors = []
    providers = []
    if cf_ready():
        providers.append(("cf", lambda: cloudflare_image(prompt, seed)))
    else:
        errors.append("cf: CF_ACCOUNT_ID or CF_API_TOKEN is not set on this server")
    if os.getenv("HF_TOKEN", "").startswith("hf_"):
        providers.append(("hf", lambda: client.text_to_image(prompt, width=768, height=768, seed=seed)))
    providers.append(("poll", lambda: pollinations_image(prompt, seed)))

    paused = [n for n, _ in providers if not _is_on(n)]
    active = [p for p in providers if p[0] not in paused]
    if not active:                      # everything paused: try all anyway
        active = providers
    else:
        for n in paused:
            errors.append(f"{n}: paused {int(_OFF[n] - time.time())}s after {_LAST.get(n, 'an error')[:90]}")

    for name, fn in active:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {str(e)[:150]}")
            print(f"{name} image failed:", e, flush=True)
            _pause(name, e)

    try:
        return local_image(short, seed)
    except Exception as e:  # ImportError on Render (no torch)
        if not isinstance(e, ImportError):
            errors.append(f"local: {str(e)[:100]}")

    # FIX 3: friendly message for users, full detail only in the server logs
    detail = " | ".join(errors) + " | " + " | ".join(_LAST.values())
    print("All image providers failed:", detail[:1500], flush=True)
    if "daily free allocation" in detail or "neurons" in detail:
        raise RuntimeError(QUOTA_MSG)
    raise RuntimeError(BUSY_MSG)


app = FastAPI(title="Comic Forge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=STORAGE), name="files")


@app.get("/api/health")
def health():
    now = time.time()
    return {"cloudflare_configured": cf_ready(),
            "cloudflare_account_ends": os.getenv("CF_ACCOUNT_ID", "").strip()[-4:],
            "hf_token_set": os.getenv("HF_TOKEN", "").startswith("hf_"),
            "paused_seconds_left": {k: max(0, int(v - now)) for k, v in _OFF.items()},
            "last_errors": _LAST}


class StoryIn(BaseModel):
    story: str = Field(min_length=20, max_length=10000)
    style: str = "Comic Book"
    panels: int = Field(6, ge=2, le=16)


def plan_story(story: str, n: int) -> dict:
    """LLM turns a story into a character sheet + panel list (strict JSON)."""
    system = (
        "You are a comic storyboard artist. Reply with ONLY JSON: "
        '{"characters":[{"name":str,"description":str}],'
        '"panels":[{"caption":str,"scene":str,"characters":[str]}]}. '
        f"Exactly {n} panels. Each character description is a FIXED visual description "
        "(age, hair, clothes, colours) under 25 words. scene = purely visual, what the camera sees. "
        "caption = max 14 words. Use character names exactly as in the character list."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": story}]
    text = None
    if cf_ready() and _is_on("cf"):     # skip Cloudflare while its quota is exhausted
        acct = os.getenv("CF_ACCOUNT_ID", "").strip()
        for model in CF_CHAT_MODELS:    # try newer models if one is deprecated (404/410)
            try:
                req = urllib.request.Request(
                    f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions",
                    data=json.dumps({"model": model, "messages": messages,
                                     "max_tokens": 3000, "temperature": 0.6}).encode(),
                    method="POST",
                    headers={"Authorization": f"Bearer {os.getenv('CF_API_TOKEN', '').strip()}",
                             "Content-Type": "application/json", "User-Agent": "comic-forge"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    text = json.loads(r.read())["choices"][0]["message"]["content"]
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:200]
                print(f"Cloudflare planning failed ({model}):", e.code, detail, flush=True)
                if e.code in (404, 410):
                    continue            # model gone: try the next one
                _pause("cf", f"HTTP {e.code}: {detail}")
                break
            except Exception as e:
                print(f"Cloudflare planning failed ({model}):", e, flush=True)
                _pause("cf", e)
                break
    if text is None:
        if not (os.getenv("HF_TOKEN", "").startswith("hf_") and _is_on("hf")):
            raise RuntimeError("no story planner available")
        try:
            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            text = client.chat_completion(model=LLM_MODEL, max_tokens=4000, temperature=0.6,
                                          messages=messages).choices[0].message.content
        except Exception as e:
            _pause("hf", e)
            raise
    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    assert data["panels"], "empty plan"
    return data


def fallback_plan(story: str, n: int) -> dict:
    """If the LLM fails, split the story into n chunks (no character sheet)."""
    sents = re.split(r"(?<=[.!?])\s+", story.strip())
    size = max(1, -(-len(sents) // n))
    chunks = [" ".join(sents[i:i + size]) for i in range(0, len(sents), size)][:n]
    hero = {"name": "Hero", "description": (sents[0] if sents else story)[:200]}
    return {"characters": [hero], "panels": [
        {"caption": " ".join(c.split()[:14]), "scene": c[:260], "characters": ["Hero"]} for c in chunks]}


def run_job(job_id: str, body: StoryIn):
    job = JOBS[job_id]
    out_dir = STORAGE / job_id
    out_dir.mkdir(exist_ok=True)
    try:
        plan = PLANS.get(job_id)        # reuse the plan on retry
        if plan is None:
            try:
                plan = plan_story(body.story, body.panels)
            except Exception as e:
                print("Story planning failed:", e, flush=True)
                plan = fallback_plan(body.story, body.panels)
            PLANS[job_id] = plan
        cast = {c["name"].lower(): c["description"] for c in plan.get("characters", [])}
        job["characters"] = plan.get("characters", [])
        job["panels"] = [{"caption": p["caption"], "scene": p["scene"], "image": None} for p in plan["panels"]]
        job["status"] = "drawing"

        client = InferenceClient(model=IMAGE_MODEL, token=os.getenv("HF_TOKEN"))
        style = STYLES.get(body.style, STYLES["Comic Book"])
        for i, p in enumerate(plan["panels"]):
            path = out_dir / f"panel_{i}.png"
            if path.exists():           # FIX 4: don't re-spend quota on finished panels
                job["panels"][i]["image"] = f"/files/{job_id}/panel_{i}.png"
                continue
            who = " ".join(f"{n}: {cast[n.lower()]}." for n in p.get("characters", []) if n.lower() in cast)
            prompt = f"{style}. {p['scene']}. {who}"
            short = f"{', '.join(style.split(',')[:2])}. {who[:150]} {p['scene'][:100]}"
            make_image(client, prompt, job["seed"], short).save(path)
            job["panels"][i]["image"] = f"/files/{job_id}/panel_{i}.png"
        job["status"] = "done"
        job.pop("error", None)
    except Exception as e:
        print("Job failed:", e, flush=True)
        job.update(status="error", error=str(e))


def build_pdf(job_id: str) -> Path:
    job, W, H, M = JOBS[job_id], 1240, 1754, 70
    cell = (W - 3 * M) // 2
    f_cap, f_title = ImageFont.load_default(size=26), ImageFont.load_default(size=64)
    pages, panels = [], job["panels"]
    for start in range(0, len(panels), 4):
        page = Image.new("RGB", (W, H), "#fffdf5"); d = ImageDraw.Draw(page)
        y0 = M
        if start == 0:
            d.text((M, M), "COMIC FORGE", font=f_title, fill="#ff3d81"); y0 = 170
        for k, p in enumerate(panels[start:start + 4]):
            x, y = M + (k % 2) * (cell + M), y0 + (k // 2) * (cell + 130)
            img = Image.open(STORAGE / job_id / f"panel_{start + k}.png").resize((cell, cell))
            page.paste(img, (x, y)); d.rectangle([x, y, x + cell, y + cell], outline="black", width=5)
            d.text((x, y + cell + 14), "\n".join(textwrap.wrap(p["caption"], 38)), font=f_cap, fill="#222")
        pages.append(page)
    path = STORAGE / job_id / "comic.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:])
    return path


@app.post("/api/stories")
def create_story(body: StoryIn):
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"id": job_id, "status": "planning", "characters": [], "panels": [],
                    "style": body.style, "seed": int.from_bytes(os.urandom(3), "big")}
    BODIES[job_id] = body
    threading.Thread(target=run_job, args=(job_id, body), daemon=True).start()
    return {"id": job_id}


@app.post("/api/stories/{job_id}/retry")
def retry_story(job_id: str):
    """Resume a failed job: finished panels are kept, only missing ones are drawn."""
    if job_id not in JOBS or job_id not in BODIES:
        raise HTTPException(404, "Story not found")
    if JOBS[job_id]["status"] != "error":
        raise HTTPException(400, "Only failed stories can be retried")
    JOBS[job_id].update(status="drawing" if job_id in PLANS else "planning")
    JOBS[job_id].pop("error", None)
    threading.Thread(target=run_job, args=(job_id, BODIES[job_id]), daemon=True).start()
    return {"id": job_id, "status": JOBS[job_id]["status"]}


@app.get("/api/stories/{job_id}")
def get_story(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Story not found")
    return JOBS[job_id]


@app.get("/api/stories/{job_id}/pdf")
def get_pdf(job_id: str):
    if job_id not in JOBS or JOBS[job_id]["status"] != "done":
        raise HTTPException(400, "Comic is not ready yet")
    return FileResponse(build_pdf(job_id), media_type="application/pdf", filename="comic.pdf")


DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():  # production: serve the built React app from the same server
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")