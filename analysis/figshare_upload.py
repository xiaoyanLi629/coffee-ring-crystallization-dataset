"""Deposit the 625-droplet release in figshare: create a private item (once), reserve its DOI, and upload every file
with figshare's chunked upload, re-sending only the parts that are not yet complete. Safe to rerun at any time; it never
publishes the item. Run on a host that can reach api.figshare.com (the AutoDL server is blocked).

Usage: python3 figshare_upload.py DIR      (DIR holds the release files; state in DIR/article.json, log DIR/upload.log)
"""
import hashlib, json, os, sys, time, urllib.error, urllib.request

API = "https://api.figshare.com/v2"
DIR = sys.argv[1]
TOK = open("/root/.figshare_token").read().strip()
STATE = DIR + "/article.json"
LOG = DIR + "/upload.log"
SKIP = {"article.json", "upload.log", "ALL_UPLOADED"}

META = {
    "title": "SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions",
    "defined_type": "dataset",
    "license": 1,  # CC BY 4.0
    "categories": [25231, 25099, 25816, 28879],  # colloid and surface chem.; analytical chem.; mineralogy and crystallography; computer vision
    "keywords": ["coffee-ring effect", "evaporative crystallisation", "scanning electron microscopy",
                 "energy-dispersive X-ray spectroscopy", "water quality", "multi-modal dataset", "distribution shift"],
    "authors": [{"name": "Xiaoyan Li"}, {"name": "Cuicui Jiang"}, {"name": "Rebecca Lahr"}],
    "description": (
        "When a drop of tap water dries, its dissolved salts crystallise into a residue whose ring, film and crystals depend on the "
        "water's composition. This record contains 625 such residues: five replicate 2 µL droplets of each of 125 synthetic waters, "
        "dried on mirror-polished 6061 aluminium at 23–26 °C and 45–50 % relative humidity. The waters form a full factorial design of "
        "NaHCO3, CaCl2 and MgSO4 (five levels each) on a constant 0.25 mM MgCl2 background. Every residue was imaged by digital optical "
        "microscopy and by scanning electron microscopy (SEM), and mapped for seven elements (C, Ca, Cl, Mg, Na, O, S) by "
        "energy-dispersive X-ray spectroscopy (EDS).<br><br>"
        "The raw instrument files are distributed in 25 archives by NaHCO3 level (Table) and block of five CaCl2 × MgSO4 combinations "
        "(Samples); extracting them into one folder rebuilds the raw/ tree. tables.zip holds the metadata table, quality flags, recipes, "
        "droplet masks, stretch-invariant EDS descriptors, SEM–EDS and optical–SEM registration, an inventory of 84,245 crystal grains, a "
        "PHREEQC equilibrium-evaporation prediction for every water and benchmark splits. README.md describes the layout and points to know "
        "before use, in particular that the SEM detector (backscattered vs secondary electrons) changed between sessions and is confounded "
        "with the NaHCO3 level. MD5SUMS and MD5SUMS_archives give checksums of the extracted files and of the archives."),
}


def log(*a):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, a)) + "\n")


def call(method, url, body=None, raw=None, timeout=120):
    if not url.startswith("http"):
        url = API + url
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "token " + TOK)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read()
    try:
        return json.loads(txt) if txt else {}
    except ValueError:  # the upload service answers part PUTs with plain text
        return {"text": txt[:200].decode(errors="replace")}


def retry(fn, *a, tries=8, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            log("HTTP", e.code, msg)
            if e.code in (400, 401, 403, 404, 409, 422):
                raise
        except Exception as e:
            log("net", type(e).__name__, str(e)[:150])
        time.sleep(min(300, 15 * 2 ** i))
    raise RuntimeError("gave up")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- item (created once) and DOI
if os.path.exists(STATE):
    st = json.load(open(STATE))
else:
    loc = retry(call, "POST", "/account/articles", META)["location"]
    st = {"id": int(loc.rstrip("/").split("/")[-1])}
    json.dump(st, open(STATE, "w"))
    log("created item", st["id"])
AID = st["id"]
if "doi" not in st:
    st["doi"] = retry(call, "POST", f"/account/articles/{AID}/reserve_doi")["doi"]
    json.dump(st, open(STATE, "w"))
    log("reserved DOI", st["doi"])

# ---- files
names = ["README.md", "MD5SUMS", "MD5SUMS_archives", "tables.zip"] + sorted(
    x for x in os.listdir(DIR) if x.startswith("raw_") and x.endswith(".zip"))
assert set(names) == {x for x in os.listdir(DIR) if x not in SKIP}, "unexpected files in DIR"
t0 = time.time()
while time.time() - t0 < 72 * 3600:
    try:
        have = {f["name"]: f for f in retry(call, "GET", f"/account/articles/{AID}/files?page_size=100")}
        todo = [n for n in names if n not in have or have[n].get("status") != "available"]
        log("remaining", len(todo), "of", len(names))
        if not todo:
            open(DIR + "/ALL_UPLOADED", "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            log("ALL_UPLOADED", st["doi"])
            break
        for n in todo:
            p = f"{DIR}/{n}"
            if n in have:
                fid = have[n]["id"]
            else:
                loc = retry(call, "POST", f"/account/articles/{AID}/files", {"name": n, "size": os.path.getsize(p), "md5": md5(p)})["location"]
                fid = int(loc.rstrip("/").split("/")[-1])
            info = retry(call, "GET", f"/account/articles/{AID}/files/{fid}")
            if info.get("status") == "available":
                continue
            up = info["upload_url"]
            parts = retry(call, "GET", up)["parts"]
            s = time.time(); sent = 0
            with open(p, "rb") as f:
                for part in parts:
                    if part["status"] == "COMPLETE":
                        continue
                    f.seek(part["startOffset"])
                    chunk = f.read(part["endOffset"] - part["startOffset"] + 1)
                    retry(call, "PUT", f"{up}/{part['partNo']}", raw=chunk, timeout=300)
                    sent += len(chunk)
            retry(call, "POST", f"/account/articles/{AID}/files/{fid}")
            log(n, "complete", f"{os.path.getsize(p) / 2**20:.1f} MB", f"{sent / 1024 / max(time.time() - s, 1e-3):.0f} KB/s")
    except Exception as e:
        log("round error", type(e).__name__, str(e)[:200])
    time.sleep(30)
else:
    log("GAVE UP after 72 h")
