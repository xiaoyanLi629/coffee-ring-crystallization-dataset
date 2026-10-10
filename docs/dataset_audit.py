import os,glob,collections,re,hashlib,json,sys
import numpy as np
from PIL import Image, ExifTags
from concurrent.futures import ThreadPoolExecutor
ROOT="/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
EL=["C","Ca","Cl","Mg","Na","O","S"]
out={}
def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f: h.update(f.read())
    return h.hexdigest()
reps=sorted(glob.glob(ROOT+"/data/Table_*/Sample_*/*_rep"))
rel=[os.path.relpath(r,ROOT+"/data") for r in reps]
out["n_reps"]=len(reps)
def one(r):
    d={"rep":r,"files":{}}
    for tree in ["data","experiment/data","experiment/data_corrected"]:
        base=os.path.join(ROOT,tree,r)
        fs=sorted(os.listdir(base)) if os.path.isdir(base) else None
        d["files"][tree]={f:(os.path.getsize(os.path.join(base,f)), md5(os.path.join(base,f))) for f in (fs or [])}
    b=os.path.join(ROOT,"data",r)
    info={}
    for f in ["sem.jpg","image.jpg"]+[e+".jpeg" for e in EL]:
        p=os.path.join(b,f)
        if not os.path.exists(p): continue
        try:
            im=Image.open(p); info[f]=[im.size,im.mode]
            if f=="image.jpg":
                ex=im.getexif(); info["exif"]={ExifTags.TAGS.get(k,k):str(v)[:40] for k,v in ex.items() if ExifTags.TAGS.get(k,k) in ("Make","Model","DateTime","Software")}
            if f.endswith(".jpeg"):
                a=np.asarray(im.convert("RGB")).astype(np.float32)
                inten=a.max(2)
                nz=a.reshape(-1,3)[inten.reshape(-1)>60]
                col=(nz.mean(0)/ (nz.mean(0).max()+1e-6)).round(2).tolist() if len(nz) else None
                info[f+"_stats"]=[float(inten.mean()),float((inten>60).mean()),col]
            if f=="sem.jpg":
                a=np.asarray(im.convert("L")).astype(np.float32)
                rows=a.mean(1); info["sem_stats"]=[float(a.mean()),float(a.std()),float(np.abs(np.diff(rows)).max())]
        except Exception as ex: info[f]="ERR "+str(ex)
    d["info"]=info
    t=os.path.join(b,"sem.txt")
    if os.path.exists(t):
        kv={}
        for line in open(t,errors="ignore"):
            m=re.match(r"\$+(\S+)\s*(.*)",line.strip())
            if m: kv[m.group(1)]=m.group(2)
        d["semtxt"]=kv
    return d
with ThreadPoolExecutor(16) as ex: res=list(ex.map(one,rel))
json.dump(res,open("/root/autodl-tmp/dataset_audit.json","w"))
print("done",len(res))
