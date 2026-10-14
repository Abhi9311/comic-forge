"""Usage (inside the backend folder):  python apply_patch2.py
Shorter provider pauses, clearer errors, and a /api/health page."""
import py_compile, shutil, sys
from pathlib import Path

PAUSE = '''_LAST = {}  # last error per provider (shown in errors and on /api/health)


def _pause(name: str, err) -> None:
    s = str(err)
    _LAST[name] = s[:200]
    if any(c in s for c in ("401", "403")):
        secs = 300       # key problem: pause 5 minutes
    elif "402" in s:
        secs = 600       # credits used up: pause 10 minutes
    elif "429" in s:
        secs = 20 if name == "cf" else 90
    else:
        secs = 0         # timeout or temporary error: try again next time
    if secs:
        _OFF[name] = time.time() + secs


'''

MAKE = '''def make_image(client, prompt: str, seed: int, short: str) -> Image.Image:
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
    active = [p for p in providers if p[0] not in paused] or providers
    if active is not providers:
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
    raise RuntimeError("Image services unavailable: " + " | ".join(errors)[:700])


'''

HEALTH = '''@app.get("/api/health")
def health():
    now = time.time()
    return {"cloudflare_configured": cf_ready(),
            "hf_token_set": os.getenv("HF_TOKEN", "").startswith("hf_"),
            "paused_seconds_left": {k: max(0, int(v - now)) for k, v in _OFF.items()},
            "last_errors": _LAST}


'''

p = Path(sys.argv[1] if len(sys.argv) > 1 else "main.py")
s = p.read_text(encoding="utf-8")
if "_LAST = {}" in s:
    raise SystemExit("Already patched. Nothing to do.")
if not Path(str(p) + ".bak2").exists():
    shutil.copy(p, str(p) + ".bak2")
a, b = s.index("def _pause("), s.index("def cf_ready(")
s = s[:a] + PAUSE + s[b:]
a, b = s.index("def make_image("), s.index("app = FastAPI(")
s = s[:a] + MAKE + s[b:]
s = s.replace("class StoryIn(BaseModel):", HEALTH + "class StoryIn(BaseModel):", 1)
p.write_text(s, encoding="utf-8")
py_compile.compile(str(p), doraise=True)
print("PATCHED OK. Backup saved as main.py.bak2")