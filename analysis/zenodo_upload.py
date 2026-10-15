"""Upload release_zenodo/ to the Zenodo draft in release_zenodo/deposition.json, retrying until every file is present
with the right MD5. Files already in the draft with a matching checksum are skipped, so the script can be rerun at any
time. It never publishes the draft. Log: /root/autodl-tmp/zen625_upload.log; marker ALL_UPLOADED when complete."""
import hashlib, json, os, time, requests
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)
ZEN = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/release_zenodo"
LOG = "/root/autodl-tmp/zen625_upload.log"
TOK = open("/root/.zenodo_token").read().strip()
H = {"Authorization": f"Bearer {TOK}"}
dep = json.load(open(ZEN + "/deposition.json"))
DID, BUCKET = dep["id"], dep["links"]["bucket"]


def log(*a):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, a)) + "\n")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def remote():
    r = requests.get(f"https://zenodo.org/api/deposit/depositions/{DID}/files", headers=H, timeout=120)
    r.raise_for_status()
    return {x["filename"]: (x["checksum"], x["id"]) for x in r.json()}


names = ["README.md", "MD5SUMS", "MD5SUMS_archives", "tables.zip"] + sorted(x for x in os.listdir(ZEN) if x.startswith("raw_") and x.endswith(".zip"))
local = {n: md5(f"{ZEN}/{n}") for n in names}
log("start", DID, len(names), "files")
t0 = time.time()
while time.time() - t0 < 72 * 3600:
    try:
        rem = remote()
    except Exception as e:
        log("list error", type(e).__name__, str(e)[:120]); time.sleep(120); continue
    todo = [n for n in names if rem.get(n, ("",))[0] != local[n]]
    log("remaining", len(todo))
    if not todo:
        open(ZEN + "/ALL_UPLOADED", "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n")); log("ALL_UPLOADED"); break
    for n in todo:
        if n in rem:  # stale or partial entry with the wrong checksum
            try:
                requests.delete(f"https://zenodo.org/api/deposit/depositions/{DID}/files/{rem[n][1]}", headers=H, timeout=120)
            except Exception as e:
                log(n, "delete error", type(e).__name__)
        size = os.path.getsize(f"{ZEN}/{n}"); s = time.time()
        try:
            with open(f"{ZEN}/{n}", "rb") as f:
                r = requests.put(f"{BUCKET}/{n}", data=f, headers={**H, "Content-Type": "application/octet-stream"}, timeout=(60, 900))
            ok = r.status_code in (200, 201) and r.json().get("checksum", "").endswith(local[n])
            log(n, r.status_code, "ok" if ok else "BAD", f"{size / 2**20:.1f} MB", f"{size / 1024 / (time.time() - s):.0f} KB/s")
        except Exception as e:
            log(n, "error", type(e).__name__, str(e)[:120]); time.sleep(30)
    time.sleep(60)
else:
    log("GAVE UP after 72 h")
