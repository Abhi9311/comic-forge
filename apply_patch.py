"""Run inside the backend folder:  python apply_patch.py
Makes the story planner use free Cloudflare AI (falls back to Hugging Face)."""
import py_compile, shutil
from pathlib import Path

NEW = '''def plan_story(story: str, n: int) -> dict:
    """LLM turns a story into a character sheet + panel list (strict JSON). Uses free Cloudflare AI first."""
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
    if cf_ready():
        try:
            acct = os.getenv("CF_ACCOUNT_ID", "").strip()
            req = urllib.request.Request(
                f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions",
                data=json.dumps({"model": "@cf/meta/llama-3.1-8b-instruct", "messages": messages,
                                 "max_tokens": 3000, "temperature": 0.6}).encode(),
                method="POST",
                headers={"Authorization": f"Bearer {os.getenv('CF_API_TOKEN', '').strip()}",
                         "Content-Type": "application/json", "User-Agent": "comic-forge"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            print("Cloudflare planning failed:", e, flush=True)
    if text is None:
        if not (os.getenv("HF_TOKEN", "").startswith("hf_") and _is_on("hf")):
            raise RuntimeError("no story planner available")
        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        text = client.chat_completion(model=LLM_MODEL, max_tokens=4000, temperature=0.6,
                                      messages=messages).choices[0].message.content
    data = json.loads(re.search(r"\\{.*\\}", text, re.S).group(0))
    assert data["panels"], "empty plan"
    return data
'''

p = Path("main.py")
s = p.read_text(encoding="utf-8")
if "Cloudflare planning failed" in s:
    raise SystemExit("Already patched. Nothing to do.")
a, b = s.index("def plan_story("), s.index("def fallback_plan(")
shutil.copy(p, "main.py.bak")
p.write_text(s[:a] + NEW + "\n\n" + s[b:], encoding="utf-8")
py_compile.compile("main.py", doraise=True)
print("PATCHED OK. Backup saved as main.py.bak")