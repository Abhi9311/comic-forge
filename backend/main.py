"""Comic Forge API: story -> LLM storyboard plan -> consistent panel images -> PDF."""
import base64, io, json, os, re, textwrap, threading, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# Load tokens from backend/.env so no terminal commands are needed (on Render, use the Environment tab)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")
_txt = Path(__file__).parent / "token.txt"  # visible alternative: just paste the token inside
if _txt.exists() and _txt.read_text().strip().startswith("hf_"):
    os.environ["HF_TOKEN"] = _txt.read_text().strip()
_tok = os.getenv("HF_TOKEN", "")
if not _tok.startswith("hf_"):
    print("\n*** HF_TOKEN missing or invalid. Open backend/token.txt and paste your token ***\n")
else:
    print("HF_TOKEN loaded OK", flush=True)
_cid = os.getenv("CF_ACCOUNT_ID", "").strip()
print(f"Cloudflare account id length={len(_cid)} ends={_cid[-4:]} token_set={bool(os.getenv('CF_API_TOKEN', '').strip())}", flush=True)

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
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

# A provider is only PAUSED for a while after an error, never switched off forever.
_OFF = {"cf": 0.0, "hf": 0.0, "poll": 0.0}


def _is_on(name: str) -> bool:
    return time.time() >= _OFF[name]


def _pause(name: str, err) -> None:
    s = str(err)
    if any(c in s for c in ("401", "403")):
        secs = 3600      # wrong key: pause for an hour
    elif "402" in s:
        secs = 1800      # credits used up: pause 30 minutes
    elif "429" in s:
        secs = 90        # rate limited: pause 90 seconds
    else:
        secs = 0         # timeout or temporary error: try again next time
    if secs:
        _OFF[name] = time.time() + secs


def cf_ready() -> bool:
    return bool(os.getenv("CF_ACCOUNT_ID", "").strip() and os.getenv("CF_API_TOKEN", "").strip())


def cloudflare_image(prompt: str, seed: int) -> Image.Image:
    """Cloudflare Workers AI (FLUX.1 schnell), free daily quota."""
    acct = os.getenv("CF_ACCOUNT_ID", "").strip()
    token = os.getenv("CF_API_TOKEN", "").strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{CF_IMAGE_MODEL}"
    body = json.dumps({"prompt": prompt[:2000], "steps": 6, "seed": seed % (2 ** 31)}).encode()
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
    """Free fallback image generator (anonymous tier: about 1 image per 15 seconds)."""
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
            if e.code != 429:
                break
            time.sleep(16)  # rate limited: wait and retry
        except Exception as e:
            last = e
            time.sleep(5)
    raise RuntimeError(f"Free image service failed: {last}")


_PIPE = {"p": None}
_LOCK = threading.Lock()


def local_image(prompt: str, seed: int) -> Image.Image:
    """Last resort: SD-Turbo running on your own PC (CPU or GPU). Not available on Render (no torch)."""
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

    # (name, function) in order of preference
    providers = []
    if cf_ready():
        providers.append(("cf", lambda: cloudflare_image(prompt, seed)))
    if os.getenv("HF_TOKEN", "").startswith("hf_"):
        providers.append(("hf", lambda: client.text_to_image(prompt, width=768, height=768, seed=seed)))
    providers.append(("poll", lambda: pollinations_image(prompt, seed)))

    # Use providers that are not paused; if all are paused, try them all anyway
    active = [p for p in providers if _is_on(p[0])] or providers

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
    raise RuntimeError("Image services unavailable: " + " | ".join(errors)[:500])


app = FastAPI(title="Comic Forge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=STORAGE), name="files")


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
    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    out = client.chat_completion(
        model=LLM_MODEL, max_tokens=4000, temperature=0.6,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": story}],
    )
    data = json.loads(re.search(r"\{.*\}", out.choices[0].message.content, re.S).group(0))
    assert data["panels"], "empty plan"
    return data


def fallback_plan(story: str, n: int) -> dict:
    """If the LLM fails, split the story into n chunks (no character sheet)."""
    sents = re.split(r"(?<=[.!?])\s+", story.strip())
    size = max(1, -(-len(sents) // n))
    chunks = [" ".join(sents[i:i + size]) for i in range(0, len(sents), size)][:n]
    # first sentence usually introduces the hero, so reuse it as the fixed character description
    hero = {"name": "Hero", "description": (sents[0] if sents else story)[:200]}
    return {"characters": [hero], "panels": [
        {"caption": " ".join(c.split()[:14]), "scene": c[:260], "characters": ["Hero"]} for c in chunks]}


def run_job(job_id: str, body: StoryIn):
    job = JOBS[job_id]
    out_dir = STORAGE / job_id; out_dir.mkdir(exist_ok=True)
    try:
        try:
            plan = plan_story(body.story, body.panels)
        except Exception as e:
            print("Story planning failed:", e, flush=True)
            _pause("hf", e)
            plan = fallback_plan(body.story, body.panels)
        cast = {c["name"].lower(): c["description"] for c in plan.get("characters", [])}
        job["characters"] = plan.get("characters", [])
        job["panels"] = [{"caption": p["caption"], "scene": p["scene"], "image": None} for p in plan["panels"]]
        job["status"] = "drawing"

        client = InferenceClient(model=IMAGE_MODEL, token=os.getenv("HF_TOKEN"))
        style = STYLES.get(body.style, STYLES["Comic Book"])
        for i, p in enumerate(plan["panels"]):
            # Consistency: same seed + the SAME character description injected into every panel
            who = " ".join(f"{n}: {cast[n.lower()]}." for n in p.get("characters", []) if n.lower() in cast)
            prompt = f"{style}. {p['scene']}. {who}"
            short = f"{', '.join(style.split(',')[:2])}. {who[:150]} {p['scene'][:100]}"
            make_image(client, prompt, job["seed"], short).save(out_dir / f"panel_{i}.png")
            job["panels"][i]["image"] = f"/files/{job_id}/panel_{i}.png"
        job["status"] = "done"
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
    threading.Thread(target=run_job, args=(job_id, body), daemon=True).start()
    return {"id": job_id}


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