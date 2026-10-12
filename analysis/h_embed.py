"""Module H (embed): frozen ImageNet ResNet18 embeddings (global average pool, 512-d) of the SEM and optical images.
SEM: grey image z-scored per image, replicated to 3 channels, 512x384.  Optical: RGB, ImageNet normalisation, 512x384.
Output derived/emb_resnet18.npz (paths, sem, opt)
"""
import numpy as np, pandas as pd, cv2, torch, torchvision
from concurrent.futures import ThreadPoolExecutor

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
dev = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
net.fc = torch.nn.Identity()
net = net.eval().to(dev)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path):
    d = f"{ROOT}/experiment/data/{path}"
    s = cv2.resize(cv2.imread(d + "/sem.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32), (512, 384), interpolation=cv2.INTER_AREA)
    s = (s - s.mean()) / (s.std() + 1e-6)
    s = np.repeat(s[None], 3, 0)
    o = cv2.resize(cv2.imread(d + "/image.jpg")[:, :, ::-1].astype(np.float32) / 255, (512, 384), interpolation=cv2.INTER_AREA)
    o = ((o - MEAN) / STD).transpose(2, 0, 1)
    return s, o


paths = pd.read_csv(ROOT + "/metadata/samples.csv").path.tolist()
with ThreadPoolExecutor(16) as ex:
    data = list(ex.map(load, paths))
out = {"sem": [], "opt": []}
with torch.no_grad():
    for i in range(0, len(paths), 16):
        for k, j in (("sem", 0), ("opt", 1)):
            x = torch.from_numpy(np.stack([d[j] for d in data[i:i + 16]])).to(dev)
            out[k].append(net(x).cpu().numpy())
np.savez_compressed(ROOT + "/derived/emb_resnet18.npz", paths=np.array(paths), sem=np.concatenate(out["sem"]), opt=np.concatenate(out["opt"]))
print("embed done", dev, np.concatenate(out["sem"]).shape)
