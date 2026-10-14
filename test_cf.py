import base64, json, urllib.request, urllib.error
acc = input("Account ID: ").strip()
tok = input("API token: ").strip()
url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/black-forest-labs/flux-1-schnell"
req = urllib.request.Request(url, data=json.dumps({"prompt": "a red apple, comic style", "steps": 4}).encode(),
    method="POST", headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
        open("test.png", "wb").write(base64.b64decode(d["result"]["image"]))
        print("WORKS: test.png saved")
except urllib.error.HTTPError as e:
    print("FAILED", e.code, e.read().decode()[:500])
