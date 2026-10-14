"""Publication figures for the Scientific Data manuscript, built from the analysis outputs (analysis/out_*).

python fig_manuscript.py [fig1 fig2 ...]   (no argument: all figures)
Outputs manuscript/figures/figN_<name>.{pdf,png}
"""
import sys, json, itertools
import numpy as np, pandas as pd, cv2
from scipy import stats
from scipy import ndimage as ndi
sys.path.insert(0, "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/analysis")
from figstyle import plt, COL1, COL2, MM, ELEMENT, SALT, SALT_TEX, DETECTOR, TABLE, OKABE, panel, save
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
A = ROOT + "/analysis"
FIG = ROOT + "/manuscript/figures"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
PAIRS = list(itertools.combinations(range(7), 2))
LEVELS = {"NaHCO3_mM": [0.25, 0.5, 0.75, 1.0, 2.0], "CaCl2_mM": [0.5, 0.75, 1.0, 1.5, 2.0], "MgSO4_mM": [0.25, 0.5, 0.75, 1.0, 2.0]}


def features():
    f = pd.read_csv(A + "/out_def/features.csv")
    f = f[f.usable_eds & f.usable_sem & (f.mask_method != "failed")].copy()
    f["recipe"] = "T" + f.table.astype(str) + "_S" + f["sample"].astype(str)
    C = np.load(A + "/out_def/coloc.npz"); ci = {p: i for i, p in enumerate(C["paths"])}
    for j, (a, b) in enumerate(PAIRS):
        f[f"x_{EL[a]}-{EL[b]}"] = [C["coloc"][ci[p]][j, 0] - C["coloc"][ci[p]][j, 1] for p in f.path]
    return f.reset_index(drop=True)


def ols_fe(y, X, groups):
    """standardised OLS slope(s) with group fixed effects; returns beta, se for the columns of X"""
    y = (y - y.mean()) / y.std()
    Xs = (X - X.mean(0)) / X.std(0)
    D = pd.get_dummies(groups, drop_first=True).values.astype(float)
    M = np.c_[np.ones(len(y)), Xs, D]
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    e = y - M @ b
    cov = np.linalg.pinv(M.T @ M) * (e @ e) / (len(y) - M.shape[1])
    k = X.shape[1]
    return b[1:k + 1], np.sqrt(np.diag(cov))[1:k + 1]


def logticks(ax, salt, axis="x"):
    lv = LEVELS[salt]
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    (ax.set_xticks if axis == "x" else ax.set_yticks)(lv)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([f"{v:g}" for v in lv])
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(mpl.ticker.NullLocator())


# ------------------------------------------------------------------ Fig. 1 design and modalities
def fig1():
    meta = pd.read_csv(ROOT + "/metadata/samples.csv")
    fig = plt.figure(figsize=(COL2, 92 * MM))
    gs = fig.add_gridspec(3, 7, height_ratios=[1.15, 1, 1], hspace=0.18, wspace=0.05, left=0.07, right=0.995, top=0.97, bottom=0.01)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(-2.2, 36); ax.set_ylim(-1.6, 7.0)
    for t, na in enumerate(LEVELS["NaHCO3_mM"]):
        x0 = 1 + t * 7
        for i in range(5):
            for j in range(5):
                ax.add_patch(Rectangle((x0 + i, j), .92, .92, fc=TABLE[t + 1], ec="white", lw=0.6, alpha=0.55 + 0.45 * (i + j) / 8))
        ax.text(x0 + 2.45, 5.25, f"Table {t + 1}", ha="center", va="bottom", fontweight="bold")
        ax.text(x0 + 2.45, 6.15, f"{SALT_TEX['NaHCO3']} {na:g} mM", ha="center", va="bottom")
    ax.annotate("", xy=(6.0, -0.45), xytext=(1.0, -0.45), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(3.5, -0.75, f"{SALT_TEX['CaCl2']} 0.5–2.0 mM", ha="center", va="top", fontsize=6.5)
    ax.annotate("", xy=(0.45, 5.0), xytext=(0.45, 0.0), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(0.15, 2.5, f"{SALT_TEX['MgSO4']}\n0.25–2.0 mM", ha="right", va="center", fontsize=6.5)
    ax.text(36, -1.5, "all waters + 0.25 mM MgCl$_2$  ·  5 replicate 2 µL droplets  ·  6061 Al, 23–26 °C, 45–50 % RH  ·  625 droplets",
            ha="right", va="bottom", fontsize=6.5, color="#444444")
    ax.text(-2.2, 7.0, "a", fontsize=8, fontweight="bold", va="top")
    for r, (path, lab) in enumerate([("Table_2/Sample_13/2_rep", "b"), ("Table_4/Sample_13/2_rep", "c")]):
        row = meta[meta.path == path].iloc[0]
        d = f"{ROOT}/experiment/data/{path}"
        o = cv2.imread(d + "/image.jpg")[:, :, ::-1].astype(np.float32)
        g = cv2.GaussianBlur(o.max(2), (0, 0), 8); ys, xs = np.nonzero(g > np.percentile(g, 80))
        cy, cx = int(np.median(ys)), int(np.median(xs)); h = int(0.33 * o.shape[0])
        o = np.clip(o[max(cy - h, 0):cy + h, max(cx - int(h * 4 / 3), 0):cx + int(h * 4 / 3)] / np.percentile(o, 99.7), 0, 1)
        sem = cv2.imread(d + "/sem.jpg", 0)
        tiles = [("Optical", o, None), (f"SEM ({row.sem_signal})", sem, "gray")]
        for e in ["Ca", "S", "Cl", "Na", "Mg"]:
            a = ndi.gaussian_filter(cv2.imread(f"{d}/{e}.jpeg").astype(np.float32).sum(2) / 765, 1.2)
            a = np.clip(a / np.percentile(a, 99.7), 0, 1)
            cm = LinearSegmentedColormap.from_list(e, ["#000000", ELEMENT[e] if e != "S" else "#F0E442", "#FFFFFF"])
            tiles.append((f"EDS {e}", a, cm))
        for k, (t, im, cm) in enumerate(tiles):
            a = fig.add_subplot(gs[r + 1, k])
            a.imshow(im, cmap=cm, interpolation="lanczos"); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True); s.set_linewidth(0.4)
            a.set_title(t, fontsize=6.5, pad=2)
            if k == 0:
                a.set_ylabel(f"T{row.table}_S{row['sample']}_R{row.replicate}\n{SALT_TEX['NaHCO3']} {row.NaHCO3_mM:g} mM", fontsize=6.5)
                a.text(-0.42, 1.08, lab, transform=a.transAxes, fontsize=8, fontweight="bold")
    save(fig, FIG + "/fig1_overview")


# ------------------------------------------------------------------ Fig. 2 acquisition schedule
def fig2():
    meta = pd.read_csv(ROOT + "/metadata/samples.csv").merge(pd.read_csv(A + "/out_g/derived.csv")[["path", "date_optical"]], on="path")
    fig, axs = plt.subplots(2, 1, figsize=(COL2, 82 * MM), gridspec_kw=dict(hspace=0.55))
    for ax, col, title in [(axs[0], "date_optical", "Optical photographs"), (axs[1], "session", "SEM/EDS sessions")]:
        d = meta.copy(); d["day"] = pd.to_datetime(d[col]); days = sorted(d.day.unique())
        for i, day in enumerate(days):
            dd = d[d.day == day]; bottom = 0
            for t in range(1, 6):
                for mode in ("BEC", "SEI"):
                    n = int(((dd.table == t) & (dd.sem_signal == mode)).sum()) if col == "session" else int((dd.table == t).sum()) * (mode == "BEC")
                    if n:
                        ax.bar(i, n, 0.72, bottom=bottom, color=TABLE[t], ec="#333333", lw=0.4, hatch="//////" if (col == "session" and mode == "SEI") else "")
                        bottom += n
        lab = [pd.Timestamp(x).strftime("%d %b") if pd.Timestamp(x).year == 2021 else pd.Timestamp(x).strftime("%d %b %Y") for x in days]
        ax.set_xticks(range(len(days))); ax.set_xticklabels(lab); ax.set_ylabel("Droplets"); ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(-0.6, len(days) - 0.4)
    mpl.rcParams["hatch.linewidth"] = 0.4
    hs = [Rectangle((0, 0), 1, 1, fc=TABLE[t], ec="#333333", lw=0.4) for t in range(1, 6)] + \
         [Rectangle((0, 0), 1, 1, fc="white", ec="#333333", lw=0.4), Rectangle((0, 0), 1, 1, fc="white", ec="#333333", lw=0.4, hatch="//////")]
    axs[1].legend(hs, [f"Table {t} ({v:g} mM)" for t, v in zip(range(1, 6), LEVELS["NaHCO3_mM"])] + ["BEC detector", "SEI detector"],
                  ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), columnspacing=1.0, handlelength=1.2)
    panel(axs[0], "a", -0.06, 1.06); panel(axs[1], "b", -0.06, 1.06)
    save(fig, FIG + "/fig2_schedule")


# ------------------------------------------------------------------ Fig. 3 EDS chemistry
def fig3():
    f = features()
    fig = plt.figure(figsize=(COL2, 118 * MM))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.25], hspace=0.55, wspace=0.42, left=0.06, right=0.99, top=0.95, bottom=0.08)
    gb = gs[1, :].subgridspec(1, 4, width_ratios=[1.3, 0.001, 1.55, 1.0], wspace=0.3)
    dose = [("Mg", "MgSO4_mM"), ("S", "MgSO4_mM"), ("O", "MgSO4_mM"), ("Cl", "CaCl2_mM"), ("Ca", "CaCl2_mM")]
    X = np.log2(f[["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]].values)
    for k, (e, s) in enumerate(dose):
        ax = fig.add_subplot(gs[0, k])
        for sess, d in f.groupby("session"):
            mm = d.groupby(s)[f"{e}_d"].mean()
            if len(mm) > 2:
                ax.plot(mm.index, mm.values, color="#B8B8B8", lw=0.5, zorder=1)
        g = f.groupby(s)[f"{e}_d"]
        med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(med.index, q1, q3, color=ELEMENT[e], alpha=0.18, lw=0, zorder=2)
        ax.plot(med.index, med.values, "o-", color=ELEMENT[e], ms=3, zorder=3)
        b, se = ols_fe(f[f"{e}_d"].values, X, f.session.values)
        j = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"].index(s)
        rho = np.median([stats.spearmanr(d[s], d[f"{e}_d"]).correlation for _, d in f.groupby("session") if d[s].nunique() > 2])
        ax.text(0.04, 0.97, f"β = {b[j]:.2f} ± {se[j]:.2f}\nρ$_{{session}}$ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=6)
        logticks(ax, s); ax.set_xlabel(f"{SALT_TEX[s[:-3]]} (mM)")
        ax.set_title(f"EDS {e}", color=ELEMENT[e], fontweight="bold")
        if k == 0:
            ax.set_ylabel("Contrast-to-noise $d$")
        panel(ax, "abcde"[k], -0.3, 1.04)
    # co-localisation matrix
    ax = fig.add_subplot(gb[0, 0])
    M = np.full((7, 7), np.nan); P = np.full((7, 7), np.nan)
    for a, b in PAIRS:
        v = f[f"x_{EL[a]}-{EL[b]}"]
        M[b, a] = v.median(); P[b, a] = (v > 0).mean()
    order = ["Na", "Cl", "Ca", "S", "O", "Mg", "C"]; oi = [EL.index(e) for e in order]
    Mo = np.full((7, 7), np.nan)
    for i in range(7):
        for j in range(7):
            a, b = oi[i], oi[j]
            v = M[max(a, b), min(a, b)] if a != b else np.nan
            if i > j:
                Mo[i, j] = v
    Mo = Mo[1:, :-1]
    im = ax.imshow(Mo, cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.2, 0.45))
    for i in range(6):
        for j in range(i + 1):
            ax.text(j, i, f"{Mo[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(Mo[i, j]) > 0.28 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(order[:-1]); ax.set_yticks(range(6)); ax.set_yticklabels(order[1:])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cax = ax.inset_axes([0.80, 0.50, 0.05, 0.46])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Excess correlation", fontsize=6); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5)
    ax.set_title("Element co-localisation", loc="left")
    panel(ax, "f", -0.12, 1.02)
    # Ca-S vs CaCl2 by NaHCO3 ; C-O vs NaHCO3
    ax = fig.add_subplot(gb[0, 2])
    for t in range(1, 6):
        d = f[f.table == t].groupby("CaCl2_mM")["x_Ca-S"].agg(["mean", "sem"])
        ax.errorbar(d.index, d["mean"], d["sem"], fmt="o-", color=TABLE[t], ms=2.5, lw=0.9, capsize=1.5, elinewidth=0.6, label=f"{LEVELS['NaHCO3_mM'][t - 1]:g}")
    logticks(ax, "CaCl2_mM"); ax.set_xlabel(f"{SALT_TEX['CaCl2']} (mM)"); ax.set_ylabel("Ca–S excess correlation")
    ax.legend(title=f"{SALT_TEX['NaHCO3']} (mM)", ncol=1, title_fontsize=6.5, loc="upper left")
    ax.set_title("Gypsum signature rises with Ca, falls with HCO$_3^-$", loc="left")
    panel(ax, "g", -0.16, 1.02)
    ax = fig.add_subplot(gb[0, 3])
    data = [f[f.NaHCO3_mM == v]["x_C-O"].values for v in LEVELS["NaHCO3_mM"]]
    bp = ax.boxplot(data, positions=range(5), widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.8),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for p_, t in zip(bp["boxes"], range(1, 6)):
        p_.set_facecolor(TABLE[t])
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{v:g}" for v in LEVELS["NaHCO3_mM"]]); ax.set_xlabel(f"{SALT_TEX['NaHCO3']} (mM)")
    ax.set_ylabel("C–O excess correlation"); ax.set_title("Carbonate signature", loc="left")
    panel(ax, "h", -0.32, 1.02)
    save(fig, FIG + "/fig3_eds_chemistry")


# ------------------------------------------------------------------ Fig. 6 EDS limitations: stretching and Al substrate
def fig6():
    pm = pd.read_csv(A + "/out_c/per_map.csv"); pm = pm[pm.usable_eds]
    f = features()
    R = np.load(A + "/out_def/radial.npz"); ri = {p: i for i, p in enumerate(R["paths"])}
    rad = np.stack([R["radial"][ri[p]] for p in f.path])
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 60 * MM), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.2, 1]))
    ax = axs[0]
    for e in EL:
        g = pm[pm.element == e]
        cv_top, cv_mean = g.p999.std() / g.p999.mean(), g["mean"].std() / g["mean"].mean()
        ax.scatter(cv_mean, cv_top, s=22, color=ELEMENT[e], ec="black", lw=0.4, zorder=3)
        ax.annotate(e, (cv_mean, cv_top), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    ax.plot([0, 0.45], [0, 0.45], ls=":", color="#888888", lw=0.7)
    ax.set_xlim(0, 0.42); ax.set_ylim(0, 0.42); ax.set_aspect("equal")
    ax.set_xlabel("CV of mean map intensity"); ax.set_ylabel("CV of 99.9th-percentile intensity")
    ax.set_title("Maps are stretched per image", loc="left"); panel(ax, "a", -0.32, 1.03)
    ax = axs[1]
    r = (np.arange(12) + 0.5) / 12
    k = EL.index("Mg")
    cmap = mpl.cm.Greens(np.linspace(0.35, 0.95, 5))
    for c_, lvl in zip(cmap, LEVELS["MgSO4_mM"]):
        sel = f.MgSO4_mM.values == lvl
        y = np.nanmedian(rad[sel, k], 0) * 100
        ax.plot(r, y, color=c_)
        ax.text(r[0] - 0.02, y[0], f"{lvl:g}", color=c_, ha="right", va="center", fontsize=6)
    ax.text(-0.02, ax.get_ylim()[1] * 1.0, f"{SALT_TEX['MgSO4']}\n(mM)", ha="right", va="top", fontsize=6)
    ax.set_xlim(-0.16, 1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Normalised radius (0 centre, 1 rim)"); ax.set_ylabel("Mg map − substrate (% of full scale)")
    ax.set_title("Rim deposits hide the substrate", loc="left"); panel(ax, "b", -0.22, 1.03)
    ax = axs[2]
    for e in ["Mg", "Na", "Ca", "S", "Cl", "O"]:
        kk = EL.index(e)
        rim = np.nanmean(rad[:, kk, -2:], 1)
        fr = [(rim[f.MgSO4_mM.values == l] < 0).mean() * 100 for l in LEVELS["MgSO4_mM"]]
        ax.plot(LEVELS["MgSO4_mM"], fr, "o-", color=ELEMENT[e], ms=2.5, label=e, lw=1.2 if e == "Mg" else 0.7)
    logticks(ax, "MgSO4_mM"); ax.set_xlabel(f"{SALT_TEX['MgSO4']} (mM)"); ax.set_ylabel("Droplets with rim below substrate (%)")
    ax.legend(ncol=2, loc="upper right"); ax.set_title("Only Mg drops below substrate", loc="left"); panel(ax, "c", -0.32, 1.03)
    save(fig, FIG + "/fig6_eds_limitations")


# ------------------------------------------------------------------ Fig. 7 acquisition structure and SEM detector
def fig7():
    vd = pd.read_csv(A + "/out_b/variance_decomposition.csv")
    vc = pd.read_csv(A + "/out_def/variance_components.csv", index_col=0)
    g = pd.read_csv(A + "/out_g/derived.csv")
    f = features(); g = g[g.path.isin(f.path)]
    G = json.load(open(A + "/out_g/summary.json"))["G3"]
    fig = plt.figure(figsize=(COL2, 118 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, left=0.07, right=0.985, top=0.95, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    fam_col = {"SEM style (raw, whole image)": OKABE["purple"], "SEM in-droplet, substrate-normalised": OKABE["vermillion"],
               "EDS in-droplet chemistry": OKABE["blue"], "EDS relative (rim ratio, d)": OKABE["sky"],
               "EDS background (outside droplet)": OKABE["grey"], "Optical global": OKABE["green"], "Geometry": OKABE["orange"]}
    for fam, d in vd.groupby("family"):
        ax.scatter(d.unique_chem, d.unique_session, s=10, color=fam_col.get(fam, "k"), ec="white", lw=0.3, label=fam, alpha=0.9)
    ax.plot([0, 0.6], [0, 0.6], ls=":", color="#888888", lw=0.7)
    ax.set_xlim(-0.01, 0.6); ax.set_ylim(-0.01, 0.6); ax.set_aspect("equal")
    ax.set_xlabel("Variance unique to chemistry"); ax.set_ylabel("Variance unique to session")
    ax.legend(loc="upper right", fontsize=5.8, handletextpad=0.2)
    ax.set_title("What each descriptor records", loc="left"); panel(ax, "a", -0.2, 1.02)
    ax = fig.add_subplot(gs[0, 1])
    feats = [("S_d", "EDS S d"), ("O_d", "EDS O d"), ("Mg_d", "EDS Mg d"), ("Ca_d", "EDS Ca d"), ("Cl_d", "EDS Cl d"), ("Na_d", "EDS Na d"),
             ("S_rbar", "EDS S radius"), ("Mg_rim_index", "Mg rim index"), ("sem_hf_in", "SEM texture"), ("sem_z_in_mean", "SEM mean grey")]
    feats = [x for x in feats if x[0] in vc.index]
    y = np.arange(len(feats))[::-1]
    left = np.zeros(len(feats))
    for comp, col, lab in [("recipe_within_session", OKABE["blue"], "Recipe"), ("session", OKABE["orange"], "Session"), ("replicate", "#CFCFCF", "Replicate")]:
        v = vc.loc[[x[0] for x in feats], comp].values
        ax.barh(y, v, left=left, color=col, ec="white", lw=0.4, height=0.7, label=lab); left += v
    ax.set_yticks(y); ax.set_yticklabels([x[1] for x in feats]); ax.set_xlim(0, 1); ax.set_xlabel("Share of variance")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); panel(ax, "b", -0.3, 1.06)
    ax = fig.add_subplot(gs[1, 0])
    order = sorted(EL, key=lambda e: {"C": 6, "O": 8, "Na": 11, "Mg": 12, "S": 16, "Cl": 17, "Ca": 20}[e])
    for i, mode in enumerate(["BEC", "SEI"]):
        d = g[g.sem_signal == mode]
        med = [d[f"semr_{e}"].sub(d[f"semr_{e}_ctrl"]).median() for e in order]
        q = [d[f"semr_{e}"].sub(d[f"semr_{e}_ctrl"]).quantile([.25, .75]).values for e in order]
        x = np.arange(7) + (i - 0.5) * 0.36
        ax.bar(x, med, 0.34, color=DETECTOR[mode], label=f"{mode} (R$^2$ = {d.sem_on_eds_R2.median():.2f})")
        ax.errorbar(x, med, yerr=np.array([[m - a, b - m] for m, (a, b) in zip(med, q)]).T, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(7)); ax.set_xticklabels([f"{e}\n{z}" for e, z in zip(order, [6, 8, 11, 12, 16, 17, 20])])
    ax.set_xlabel("Element (atomic number)"); ax.set_ylabel("SEM grey–EDS excess correlation")
    ax.legend(loc="lower left"); ax.set_title("BEC images composition, SEI images topography", loc="left"); panel(ax, "c", -0.2, 1.02)
    ax = fig.add_subplot(gs[1, 1])
    names = {"sem_z_in_mean": "Mean grey", "sem_z_in_sd": "Grey s.d.", "sem_hf_in": "Texture", "sem_hf_rim": "Rim texture",
             "sem_bright_frac": "Bright fraction", "sem_dark_frac": "Dark fraction", "sem_rim_width": "Rim width"}
    keys = list(names)
    yy = np.arange(len(keys))[::-1]
    sets = [("BEC − SEI at fixed NaHCO$_3$", [G["mode_effect_T2T3"][k] for k in keys], OKABE["purple"], -0.22),
            ("NaHCO$_3$ within BEC", [G["NaHCO3_within_mode_sessionFE"][k]["BEC"] for k in keys], DETECTOR["BEC"], 0.0),
            ("NaHCO$_3$ within SEI", [G["NaHCO3_within_mode_sessionFE"][k]["SEI"] for k in keys], DETECTOR["SEI"], 0.22)]
    for lab, vals, col, off in sets:
        b = np.array([v[0] for v in vals]); se = np.array([v[1] for v in vals])
        ax.errorbar(b, yy + off, xerr=1.96 * se, fmt="o", color=col, ms=3, elinewidth=0.8, capsize=1.5, label=lab)
    ax.axvline(0, color="black", lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in keys])
    ax.set_xlabel("Effect (s.d. units; ±95% CI)"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=6, columnspacing=0.8)
    ax.set_title("Separating detector from chemistry", loc="left"); panel(ax, "d", -0.3, 1.02)
    save(fig, FIG + "/fig7_acquisition")


# ------------------------------------------------------------------ Fig. 8 optical modality
def fig8():
    g = pd.read_csv(A + "/out_g/derived.csv"); f = features(); g = g[g.path.isin(f.path) & (g.opt_mask_method != "failed")]
    GS = json.load(open(A + "/out_g/summary.json"))
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 58 * MM), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    for w, d in g.groupby("opt_w"):
        lab = "2592×1944" if w == 2592 else "1280×1024"
        ax.scatter(d.eds_diam_px_half * 2, d.opt_diam_px, s=5, alpha=0.7, color=OKABE["blue"] if w == 2592 else OKABE["orange"], lw=0, label=lab)
    d = g[g.opt_w == 2592]; r = np.corrcoef(d.eds_diam_px_half, d.opt_diam_px)[0, 1]
    ax.text(0.04, 0.96, f"r = {r:.2f} (2592×1944)", transform=ax.transAxes, va="top")
    ax.set_xlabel("EDS droplet diameter (EDS px)"); ax.set_ylabel("Optical droplet diameter (px)"); ax.legend(loc="lower right", title="Camera setting", title_fontsize=6.5)
    ax.set_title("Droplet size agrees", loc="left"); panel(ax, "a", -0.3, 1.03)
    ax = axs[1]
    eff_opt = GS["G5"]["salt_effects_optical_dateFE"]["opt_diam_mm_rel"]
    e3 = json.load(open(A + "/out_def/summary.json"))["E4_size"]["vs_salts"]["session_FE"]
    keys = [("lNa", "NaHCO3"), ("lCa", "CaCl2"), ("lMg", "MgSO4")]
    for i, (kk, s) in enumerate(keys):
        bo = eff_opt[kk]
        ax.errorbar(bo[0], i + 0.12, xerr=1.96 * bo[1], fmt="s", color=OKABE["green"], ms=3, capsize=1.5, label="Optical (date FE)" if i == 0 else None)
        be = e3[f"log_{s}_mM"]
        if True:
            ax.errorbar(be[0], i - 0.12, xerr=1.96 * be[1], fmt="o", color=OKABE["blue"], ms=3, capsize=1.5, label="EDS mask (session FE)" if i == 0 else None)
    ax.axvline(0, color="black", lw=0.6); ax.set_yticks(range(3)); ax.set_yticklabels([SALT_TEX[s] for _, s in keys]); ax.invert_yaxis()
    ax.set_xlabel("Effect on deposit diameter (s.d. per s.d.)"); ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Two instruments, one size response", loc="left"); panel(ax, "b", -0.3, 1.03)
    ax = axs[2]
    ok = g.optreg_ncc_ctrl.notna()
    bins = np.linspace(0, 0.8, 33)
    ax.hist(g.optreg_ncc[ok], bins, color=OKABE["blue"], alpha=0.75, label="Same droplet")
    ax.hist(g.optreg_ncc_ctrl[ok], bins, color=OKABE["grey"], alpha=0.6, label="Other replicate")
    ax.set_xlabel("Optical–SEM gradient correlation"); ax.set_ylabel("Droplets"); ax.legend()
    ax.set_title("Crystal-level correspondence is weak", loc="left"); panel(ax, "c", -0.3, 1.03)
    save(fig, FIG + "/fig8_optical")


# ------------------------------------------------------------------ Fig. 4 spatial organisation and grain inventory
PHASE_COL = {"gypsum": "#D55E00", "Ca carbonate": "#56B4E9", "Mg sulfate": "#009E73", "halite": "#E69F00",
             "chloride brine": "#CC79A7", "Na/Mg carbonate": "#9E9E9E"}


def fig4():
    PM = np.load(A + "/out_p/phase_maps.npz", allow_pickle=True)
    names = list(PM["names"]); pi = {p: i for i, p in enumerate(PM["paths"])}
    ph = pd.read_csv(A + "/out_p/phase_droplet.csv"); ph = ph[ph.usable_eds]
    meta = pd.read_csv(ROOT + "/metadata/samples.csv")
    fig = plt.figure(figsize=(COL2, 108 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.15], hspace=0.22, left=0.06, right=0.99, top=0.95, bottom=0.09)
    g1 = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.55], wspace=0.06)
    ex = ["Table_1/Sample_25/1_rep", "Table_2/Sample_13/2_rep", "Table_4/Sample_13/2_rep", "Table_5/Sample_25/1_rep"]
    for k, path in enumerate(ex):
        ax = fig.add_subplot(g1[0, k])
        row = meta[meta.path == path].iloc[0]
        sem = cv2.resize(cv2.imread(f"{ROOT}/experiment/data/{path}/sem.jpg", 0), (505, 379), interpolation=cv2.INTER_AREA).astype(np.float32)
        sem = (sem - np.percentile(sem, 1)) / (np.percentile(sem, 99) - np.percentile(sem, 1))
        rgb = np.repeat(np.clip(sem, 0, 1)[..., None] * 0.55, 3, -1)
        lab = PM["labels"][pi[path]]
        for j, nm in enumerate(names):
            c_ = np.array(mpl.colors.to_rgb(PHASE_COL[nm]))
            rgb[lab == j] = c_
        ax.imshow(rgb, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.4)
        ax.set_title(f"{SALT_TEX['NaHCO3']} {row.NaHCO3_mM:g} · {SALT_TEX['CaCl2']} {row.CaCl2_mM:g} · {SALT_TEX['MgSO4']} {row.MgSO4_mM:g}", fontsize=6)
        if k == 0:
            panel(ax, "a", -0.08, 1.1)
    ax = fig.add_subplot(g1[0, 4]); ax.axis("off")
    for j, nm in enumerate(["gypsum", "Ca carbonate", "Mg sulfate", "halite", "chloride brine", "Na/Mg carbonate"]):
        ax.add_patch(Rectangle((0.02, 0.86 - j * 0.14), 0.14, 0.09, color=PHASE_COL[nm], transform=ax.transAxes))
        ax.text(0.22, 0.905 - j * 0.14, nm.replace("Na/Mg", "Na/Mg"), transform=ax.transAxes, va="center", fontsize=6.5)
    ax.text(0.02, 0.98, "Grain phase", transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="bottom")
    g2 = gs[1].subgridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.42)
    f = features()
    R = np.load(A + "/out_def/radial.npz"); ri = {p: i for i, p in enumerate(R["paths"])}
    rs = np.stack([R["radial_shape"][ri[p]] for p in f.path])
    ax = fig.add_subplot(g2[0])
    r = (np.arange(12) + 0.5) / 12
    for e in ["S", "Ca", "Cl", "O", "Na", "C", "Mg"]:
        k = EL.index(e)
        ax.plot(r, np.nanmedian(rs[:, k], 0), color=ELEMENT[e], lw=1.1, label=e)
    ax.axhline(1, color="#999999", lw=0.5, ls=":")
    ax.set_xlabel("Normalised radius (0 centre, 1 rim)"); ax.set_ylabel("Element signal / droplet mean")
    ax.legend(ncol=2, loc="upper left"); ax.set_title("Element radial profiles (median)", loc="left"); panel(ax, "b", -0.25, 1.03)
    ax = fig.add_subplot(g2[1])
    onset = pd.read_csv(A + "/out_t/onset.csv")
    order = ["gypsum", "Ca carbonate", "Mg sulfate", "halite", "chloride brine", "Na/Mg carbonate"]
    data = [ph[f"r_{nm}"].dropna().values for nm in order]
    bp = ax.boxplot(data, vert=False, positions=range(len(order))[::-1], widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.5))
    for p_, nm in zip(bp["boxes"], order):
        p_.set_facecolor(PHASE_COL[nm])
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels(order)
    ax.set_xlabel("Grain position (normalised radius)"); ax.set_xlim(0.2, 1.0)
    cf = {"gypsum": onset.Gypsum_onset_cf.median(), "Ca carbonate": onset.Calcite_onset_cf.median(),
          "Mg sulfate": onset[["Epsomite_onset_cf", "Kieserite_onset_cf", "Bloedite_onset_cf"]].min(1).median(),
          "halite": onset.Halite_onset_cf.median(), "chloride brine": np.nan, "Na/Mg carbonate": onset.Nesquehonite_onset_cf.median()}
    for yy, nm in zip(range(len(order))[::-1], order):
        txt = "stays dissolved" if np.isnan(cf[nm]) else f"CF {cf[nm]:,.0f}"
        ax.text(0.215, yy + 0.33, txt, fontsize=5.3, color="#555555")
    ax.set_title("Grain position vs predicted onset", loc="left"); panel(ax, "c", -0.42, 1.03)
    ax = fig.add_subplot(g2[2])
    grid = ph.groupby(["NaHCO3_mM", "CaCl2_mM"])["area_gypsum"].mean().unstack() * 100
    im = ax.imshow(grid.values, origin="lower", cmap="Oranges", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{v:g}" for v in grid.columns]); ax.set_yticks(range(5)); ax.set_yticklabels([f"{v:g}" for v in grid.index])
    ax.set_xlabel(f"{SALT_TEX['CaCl2']} (mM)"); ax.set_ylabel(f"{SALT_TEX['NaHCO3']} (mM)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{grid.values[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color="white" if grid.values[i, j] > np.nanpercentile(grid.values, 70) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Gypsum grains (% of area)", fontsize=6); cb.outline.set_linewidth(0.4)
    ax.set_title("Gypsum needs Ca left over from carbonate", loc="left"); panel(ax, "d", -0.3, 1.03)
    save(fig, FIG + "/fig4_spatial_phases")


# ------------------------------------------------------------------ Fig. 5 thermodynamic model vs observation
def fig5():
    paths = pd.read_csv(A + "/out_t/evap_paths.csv")
    cmp_ = pd.read_csv(A + "/out_t/compare.csv", index_col=0)
    meta = pd.read_csv(ROOT + "/metadata/samples.csv").drop_duplicates(["table", "sample"])
    meta["recipe"] = "T" + meta.table.astype(str) + "_S" + meta["sample"].astype(str)
    tot = meta.set_index("recipe")[["Ca_mM", "Mg_mM", "Na_mM", "CaCl2_mM", "MgSO4_mM", "NaHCO3_mM"]]
    fig = plt.figure(figsize=(COL2, 112 * MM))
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.2, left=0.06, right=0.99, top=0.94, bottom=0.09)
    ax = fig.add_subplot(gs[0, :3])
    rec = "T3_S13"; d = paths[paths.recipe == rec]; t = tot.loc[rec]
    cf = 1 / d.mass_H2O.values; aw = 10 ** d.la_H2O.values
    col = lambda m: d[m].values if m in d else np.zeros(len(d))
    curves = [("Calcite (Ca)", col("Calcite") / (t.Ca_mM / 1e3), OKABE["grey"]),
              ("Gypsum (Ca)", col("Gypsum") / (t.Ca_mM / 1e3), ELEMENT["Ca"]),
              ("Mg sulfates (Mg)", (col("Epsomite") + col("Hexahydrite") + col("Kieserite") + col("Bloedite")) / (t.Mg_mM / 1e3), ELEMENT["Mg"]),
              ("Nesquehonite (Mg)", col("Nesquehonite") / (t.Mg_mM / 1e3), "#7FC97F"),
              ("Halite (Na)", col("Halite") / (t.Na_mM / 1e3), ELEMENT["Na"])]
    for lab, y, c_ in curves:
        ax.plot(cf, np.clip(y, 0, 1.05), color=c_, lw=1.2, label=lab)
    i_dry = np.nonzero(aw >= 0.475)[0][-1]
    ax.axvspan(cf[i_dry], cf.max() * 1.3, color="#EEEEEE", zorder=0)
    ax.text(cf[i_dry] * 1.1, 0.5, "below the drying RH:\nbrine film remains", fontsize=6, color="#555555", va="center")
    ax.set_xscale("log"); ax.set_xlim(1, cf.max() * 1.3); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Concentration factor (initial / remaining water)"); ax.set_ylabel("Fraction of element precipitated")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title(f"Evaporation path, {rec.replace('_', ' ')} ({SALT_TEX['NaHCO3']} {t.NaHCO3_mM:g}, {SALT_TEX['CaCl2']} {t.CaCl2_mM:g}, {SALT_TEX['MgSO4']} {t.MgSO4_mM:g} mM)", loc="left")
    panel(ax, "a", -0.12, 1.03)
    ax = fig.add_subplot(gs[0, 3:])
    sub = cmp_[(cmp_.CaCl2_mM == 1.0) & (cmp_.MgSO4_mM == 0.75)].sort_values("NaHCO3_mM")
    x = np.arange(len(sub))
    cal, gyp = sub.f_Ca_calcite.clip(0, 1).values, sub.f_Ca_gypsum.clip(0, 1).values
    ax.bar(x, cal, 0.65, color=OKABE["grey"], label="Calcite")
    ax.bar(x, gyp, 0.65, bottom=cal, color=ELEMENT["Ca"], label="Gypsum")
    ax.bar(x, np.clip(1 - cal - gyp, 0, 1), 0.65, bottom=cal + gyp, color="#DDDDDD", label="Dissolved in brine")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in sub.NaHCO3_mM]); ax.set_xlabel(f"{SALT_TEX['NaHCO3']} (mM)")
    ax.set_ylabel("Fraction of Ca at the drying endpoint"); ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3)
    panel(ax, "b", -0.14, 1.03)
    ph = pd.read_csv(A + "/out_p/phase_droplet.csv"); ph = ph[ph.usable_eds]
    ph["recipe"] = "T" + ph.table.astype(str) + "_S" + ph["sample"].astype(str)
    cmp_ = cmp_.join(ph.groupby("recipe")[["area_gypsum", "area_Mg sulfate", "area_halite"]].mean() * 100, how="inner")
    tests = [("Gypsum_mmol", "area_gypsum", "Predicted gypsum (mmol kg$^{-1}$ water)", "Gypsum grains (% of droplet area)"),
             ("MgSO4_solid_mmol", "area_Mg sulfate", "Predicted Mg-sulfate solids (mmol kg$^{-1}$)", "Mg-sulfate grains (% of droplet area)"),
             ("Halite_mmol", "area_halite", "Predicted halite (mmol kg$^{-1}$)", "Halite grains (% of droplet area)")]
    for k, (p, o, xl, yl) in enumerate(tests):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        for tb in range(1, 6):
            dd = cmp_[cmp_.table == tb]
            ax.scatter(dd[p], dd[o], s=9, color=TABLE[tb], ec="#555555", lw=0.25, label=f"{LEVELS['NaHCO3_mM'][tb - 1]:g}")
        within = [stats.spearmanr(g_[p], g_[o]).correlation for _, g_ in cmp_.groupby("table")]
        ax.text(0.03, 0.97, f"ρ$_{{all}}$ = {stats.spearmanr(cmp_[p], cmp_[o]).correlation:.2f}\nρ$_{{within\\,Table}}$ = {np.median(within):.2f}",
                transform=ax.transAxes, va="top", fontsize=6)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        if k == 1:
            ax.legend(title=f"{SALT_TEX['NaHCO3']} (mM)", title_fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
                      fontsize=6, handletextpad=0.1, columnspacing=0.8)
        panel(ax, "cde"[k], -0.3, 1.03)
    save(fig, FIG + "/fig5_thermodynamics")


# ------------------------------------------------------------------ Fig. 9 benchmarks
def fig9():
    lin = pd.read_csv(A + "/out_h/h1_table.csv")
    cnn = pd.read_csv(A + "/out_m/summary.csv")
    rows = [("Detector label only", "lin", "Detector mode only"), ("EDS descriptors (ridge)", "lin", "EDS (hand-crafted)"),
            ("SEM ResNet-18, frozen (ridge)", "lin", "SEM (ResNet18)"), ("Optical ResNet-18, frozen (ridge)", "lin", "Optical (ResNet18)"),
            ("EDS CNN, fine-tuned", "cnn", "eds"), ("EDS + SEM CNN, fine-tuned", "cnn", "eds_sem"),
            ("SEM CNN, fine-tuned", "cnn", "sem"), ("Optical CNN, fine-tuned", "cnn", "opt")]
    salts = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]
    prots = [("P1", "P1 recipe CV"), ("P2", "P2 leave-one-session-out")]
    M = np.full((len(rows), 6), np.nan)
    for i, (_, src, key) in enumerate(rows):
        for j, s in enumerate(salts):
            for q, (pc, pl) in enumerate(prots):
                if src == "lin":
                    v = lin[(lin.input == key) & (lin.target == s) & (lin.protocol == pl)].R2
                else:
                    v = cnn[(cnn.input == key) & (cnn.target == s) & (cnn.protocol == pc)].R2
                if len(v):
                    M[i, 2 * j + q] = v.iloc[0]
    fig = plt.figure(figsize=(COL2, 92 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.05, 1.05], wspace=0.45, left=0.2, right=0.99, top=0.88, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    cmap = mpl.cm.YlGnBu.copy(); cmap.set_under("#F2F2F2")
    im = ax.imshow(np.where(np.isnan(M), -9, M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if M[i, j] > 0.6 else "black")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(range(6)); ax.set_xticklabels(["Recipe\nCV", "Session\nout"] * 3, fontsize=6)
    for j, s in enumerate(salts):
        ax.text(2 * j + 0.5, -0.95, SALT_TEX[s[:-3]], ha="center", va="bottom", fontsize=7, fontweight="bold")
    for xx in (1.5, 3.5):
        ax.axvline(xx, color="white", lw=2)
    ax.axhline(3.5, color="white", lw=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    panel(ax, "a", -0.62, 1.08)
    # cross-detector transfer (fine-tuned CNNs)
    ax = fig.add_subplot(gs[1])
    keys = [("eds", "EDS"), ("eds_sem", "EDS+SEM"), ("sem", "SEM"), ("opt", "Optical")]
    for j, s in enumerate(["CaCl2_mM", "MgSO4_mM"]):
        v = [cnn[(cnn.input == k) & (cnn.target == s) & (cnn.protocol == "P3")].R2 for k, _ in keys]
        v = np.array([x.iloc[0] if len(x) else np.nan for x in v])
        xs = np.arange(len(keys)) + (j - 0.5) * 0.36
        ax.bar(xs, np.clip(v, -1, 1), 0.34, color=SALT[s[:-3]], label=SALT_TEX[s[:-3]])
        for xx, vv in zip(xs, v):
            if vv < -1:
                ax.text(xx, -1.02, f"{vv:.1f}", rotation=90, ha="center", va="top", fontsize=5)
    ax.axhline(0, color="black", lw=0.6); ax.set_ylim(-1.35, 1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys], rotation=30, ha="right")
    ax.set_ylabel("R$^2$, train one detector, test the other"); ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Cross-detector transfer", loc="left"); panel(ax, "b", -0.42, 1.08)
    # recipe-level predictions of the EDS CNN, sessions held out
    ax = fig.add_subplot(gs[2])
    p = pd.read_csv(A + "/out_m/preds_eds_P2.csv")
    for s in salts:
        r = p.groupby(["table", "sample"])[[f"true_{s}", f"pred_{s}"]].mean()
        ax.scatter(2 ** r[f"true_{s}"] * np.exp(np.random.default_rng(1).normal(0, 0.03, len(r))), 2 ** r[f"pred_{s}"], s=6,
                   color=SALT[s[:-3]], alpha=0.8, lw=0, label=SALT_TEX[s[:-3]])
    ax.plot([0.2, 2.5], [0.2, 2.5], ls=":", color="#888888", lw=0.7)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    tk = [0.25, 0.5, 1, 2]
    ax.set_xticks(tk); ax.set_xticklabels([f"{v:g}" for v in tk]); ax.set_yticks(tk); ax.set_yticklabels([f"{v:g}" for v in tk])
    ax.set_xlabel("True concentration (mM)"); ax.set_ylabel("Predicted, recipe mean (mM)")
    ax.legend(loc="upper left", fontsize=6); ax.set_title("EDS CNN, sessions held out", loc="left"); panel(ax, "c", -0.42, 1.08)
    save(fig, FIG + "/fig9_benchmarks")


ALL = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9}
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
        print("done", k, flush=True)
