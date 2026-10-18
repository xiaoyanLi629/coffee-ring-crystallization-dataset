"""Module M (deep): fine-tuned CNN baselines for image -> concentration, under the dataset's evaluation protocols.

Inputs (320x240, aspect kept): SEM (grey, z-scored per image), optical (RGB, ImageNet-normalised), EDS (7 channels,
decoded (R+G+B)/765, each map divided by its own 99.5th percentile, i.e. used as relative maps), and EDS+SEM (8 ch).
Model: ImageNet ResNet-18, first convolution adapted to the channel count (pretrained RGB filters averaged and tiled),
3 regression outputs (standardised log2 NaHCO3, CaCl2, MgSO4), MSE loss, AdamW, cosine schedule, 30 epochs,
augmentation by the 8 dihedral transforms (the residues have no preferred orientation) and mild intensity jitter.
Protocols: P1 recipe-grouped 5-fold (metadata cv_test_fold), P2 leave-one-SEM-session-out (12 folds), P3 cross-detector.
Test-time augmentation: mean over the 8 dihedral transforms.
Outputs analysis/out_m/{preds_<input>_<protocol>.csv, summary.json}; cache derived/deep_cache.npz
Usage: python m_deep.py [cache|train INPUT PROTOCOL|summary]
"""
import os, sys, json, time
import numpy as np, pandas as pd, cv2

ROOT = os.environ.get("CRD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root; override with CRD_ROOT
OUT = ROOT + "/analysis/out_m"
CACHE = ROOT + "/derived/deep_cache.npz"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
SALT = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]
W, H = 320, 240


def droplets():
    m = pd.read_csv(ROOT + "/metadata/samples.csv")
    failed = set(pd.read_csv(ROOT + "/analysis/out_def/features.csv").query("mask_method == 'failed'").path)
    m = m[m.usable_eds & m.usable_sem & ~m.path.isin(failed)].reset_index(drop=True)
    return m


def load_one(path):
    d = f"{ROOT}/experiment/data/{path}"
    sem = cv2.resize(cv2.imread(d + "/sem.jpg", 0), (W, H), interpolation=cv2.INTER_AREA)
    opt = np.ascontiguousarray(np.rot90(cv2.imread(d + "/image.jpg"), -1)[:, :, ::-1])  # photographs are distributed in the SEM/EDS orientation; turn back to the acquisition orientation used for the published results
    opt = cv2.resize(opt, (W, H), interpolation=cv2.INTER_AREA)
    eds = []
    for e in EL:
        a = cv2.imread(f"{d}/{e}.jpeg").astype(np.float32).sum(2) / 765
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        eds.append(np.clip(a / (np.percentile(a, 99.5) + 1e-6), 0, 1.5))
    return sem, opt, np.stack(eds).astype(np.float16)


def build_cache():
    from concurrent.futures import ThreadPoolExecutor
    m = droplets()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(load_one, m.path))
    np.savez(CACHE, paths=m.path.values, sem=np.stack([r[0] for r in res]), opt=np.stack([r[1] for r in res]),
             eds=np.stack([r[2] for r in res]))
    print("cache", len(m))


def make_input(c, name):
    sem = c["sem"].astype(np.float32)
    sem = (sem - sem.mean((1, 2), keepdims=True)) / (sem.std((1, 2), keepdims=True) + 1e-6)
    if name == "sem":
        return sem[:, None]
    if name == "opt":
        o = c["opt"].astype(np.float32) / 255
        o = (o - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return o.transpose(0, 3, 1, 2).astype(np.float32)
    eds = (c["eds"].astype(np.float32) - 0.25) / 0.25
    if name == "eds":
        return eds
    if name == "eds_sem":
        return np.concatenate([eds, sem[:, None]], 1)
    raise ValueError(name)


def splits(m, protocol):
    idx = np.arange(len(m))
    if protocol == "P1":
        return [(f"fold{k}", idx[m.cv_test_fold != k], idx[m.cv_test_fold == k]) for k in sorted(m.cv_test_fold.unique())]
    if protocol == "P2":
        return [(s, idx[m.session != s], idx[m.session == s]) for s in sorted(m.session.unique())]
    if protocol == "P3":
        b = (m.sem_signal == "BEC").values
        return [("BEC_to_SEI", idx[b], idx[~b]), ("SEI_to_BEC", idx[~b], idx[b])]
    raise ValueError(protocol)


def net(cin):
    import torch, torchvision
    n = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    if cin != 3:
        w = n.conv1.weight.data.mean(1, keepdim=True)
        n.conv1 = torch.nn.Conv2d(cin, 64, 7, 2, 3, bias=False)
        n.conv1.weight.data = w.repeat(1, cin, 1, 1) * 3 / cin
    n.fc = torch.nn.Linear(512, 3)
    return n


def dihedral(x, k):
    import torch
    x = torch.rot90(x, k % 4, (2, 3))
    return torch.flip(x, (3,)) if k >= 4 else x


def train_eval(X, Y, tr, te, dev, epochs=30, bs=32, seed=0):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Y[tr].mean(0), Y[tr].std(0)
    Yt = torch.tensor((Y - mu) / sd, dtype=torch.float32)
    Xt = torch.tensor(X)
    model = net(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = Xt[b].to(dev, non_blocking=True)
            x = dihedral(x, np.random.randint(8))
            x = x * (1 + 0.1 * torch.randn(len(b), 1, 1, 1, device=dev)) + 0.1 * torch.randn(len(b), 1, 1, 1, device=dev)
            loss = torch.nn.functional.mse_loss(model(x), Yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            x = Xt[te[i:i + 64]].to(dev)
            out.append(torch.stack([model(dihedral(x, k)) for k in range(8)]).mean(0).cpu().numpy())
    return np.concatenate(out) * sd + mu


def train(name, protocol):
    import torch
    os.makedirs(OUT, exist_ok=True)
    dev = os.environ.get("DEV", "cuda:1")
    m = droplets()
    c = np.load(CACHE, allow_pickle=True)
    assert (c["paths"] == m.path.values).all()
    X = make_input(c, name).astype(np.float32)
    Y = np.log2(m[SALT].values)
    pred = np.full_like(Y, np.nan)
    t0 = time.time()
    for fold, tr, te in splits(m, protocol):
        pred[te] = train_eval(X, Y, tr, te, dev)
        print(name, protocol, fold, f"{time.time() - t0:.0f}s", flush=True)
    df = m[["droplet_id", "path", "table", "sample", "session", "sem_signal"]].copy()
    for j, s in enumerate(SALT):
        df[f"true_{s}"] = Y[:, j]; df[f"pred_{s}"] = pred[:, j]
    df.to_csv(f"{OUT}/preds_{name}_{protocol}.csv", index=False)


def summary():
    from sklearn.metrics import r2_score
    rows = []
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("preds_"):
            continue
        name, protocol = f[6:-4].rsplit("_", 1)
        d = pd.read_csv(f"{OUT}/{f}")
        for s in SALT:
            dd = d.dropna(subset=[f"pred_{s}"])
            if protocol == "P3" and s == "NaHCO3_mM":
                dd = dd[dd.table.isin([2, 3])]
            rec = dd.groupby(["table", "sample"])[[f"true_{s}", f"pred_{s}"]].mean()
            rows.append({"input": name, "protocol": protocol, "target": s, "n": len(dd),
                         "R2": round(r2_score(dd[f"true_{s}"], dd[f"pred_{s}"]), 3),
                         "R2_recipe": round(r2_score(rec[f"true_{s}"], rec[f"pred_{s}"]), 3),
                         "MAE_log2": round(float(np.abs(dd[f"true_{s}"] - dd[f"pred_{s}"]).mean()), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT + "/summary.csv", index=False)
    print(t.pivot_table(index=["target", "input"], columns="protocol", values="R2").round(2).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        build_cache()
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3])
    else:
        summary()
