"""Figs. 2-8 of the manuscript, built from the analysis outputs (analysis/out_*). Fig. 1 is fig1.py.

python figs.py [fig2 fig3 ...]     (no argument: all)  ->  manuscript/figures/figN_<name>.{pdf,png}

Every panel is placed in millimetres (style.axes_mm), so widths, margins and panel letters are identical across figures.
Panels carry no headline titles; what a panel shows is stated by its axis labels and the caption.
"""
import sys, json, itertools
import numpy as np, pandas as pd, cv2
from scipy import stats
sys.path.insert(0, "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/analysis/figures")
from style import (plt, figure, axes_mm, letter, save, ELEMENT, SALT, SALT_TEX, DETECTOR, TABLE, GREY, LIGHT, INK)
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
A = ROOT + "/analysis"
FIG = ROOT + "/manuscript/figures"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
PAIRS = list(itertools.combinations(range(7), 2))
LEVELS = {"NaHCO3_mM": [0.25, 0.5, 0.75, 1.0, 2.0], "CaCl2_mM": [0.5, 0.75, 1.0, 1.5, 2.0], "MgSO4_mM": [0.25, 0.5, 0.75, 1.0, 2.0]}
PHASE_COL = {"gypsum": "#D9622B", "Ca carbonate": "#4BA3D9", "Mg sulfate": "#B655BE", "halite": "#F2B134",
             "chloride brine": "#7FBF7B", "Na/Mg carbonate": "#A6A6A6"}
PHASES = ["gypsum", "Ca carbonate", "Mg sulfate", "halite", "chloride brine", "Na/Mg carbonate"]


def features():
    f = pd.read_csv(A + "/out_def/features.csv")
    f = f[f.usable_eds & f.usable_sem & (f.mask_method != "failed")].copy()
    f["recipe"] = "T" + f.table.astype(str) + "_S" + f["sample"].astype(str)
    C = np.load(A + "/out_def/coloc.npz"); ci = {p: i for i, p in enumerate(C["paths"])}
    for j, (a, b) in enumerate(PAIRS):
        f[f"x_{EL[a]}-{EL[b]}"] = [C["coloc"][ci[p]][j, 0] - C["coloc"][ci[p]][j, 1] for p in f.path]
    return f.reset_index(drop=True)


def ols_fe(y, X, groups):
    """standardised OLS slopes with group fixed effects; returns beta and s.e. for the columns of X"""
    y = (y - y.mean()) / y.std()
    Xs = (X - X.mean(0)) / X.std(0)
    D = pd.get_dummies(groups, drop_first=True).values.astype(float)
    M = np.c_[np.ones(len(y)), Xs, D]
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    e = y - M @ b
    cov = np.linalg.pinv(M.T @ M) * (e @ e) / (len(y) - M.shape[1])
    k = X.shape[1]
    return b[1:k + 1], np.sqrt(np.diag(cov))[1:k + 1]


def logx(ax, salt, axis="x"):
    lv = LEVELS[salt]
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    (ax.set_xticks if axis == "x" else ax.set_yticks)(lv)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([f"{v:g}" for v in lv])
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(mpl.ticker.NullLocator())


def mm_letter(fig, ax, s, dx=-9, dy=-3):
    """panel letter dx mm left of and dy mm above the axes' top-left corner"""
    W, H = fig.get_size_inches() * 25.4
    p = ax.get_position()
    letter(fig, p.x0 * W + dx, (1 - p.y1) * H + dy, s)


def tidy_image(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ Fig. 2 EDS response to the design
def fig2():
    f = features()
    fig = figure(180, 104)
    X = np.log2(f[["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]].values)
    dose = [("Mg", "MgSO4_mM"), ("S", "MgSO4_mM"), ("O", "MgSO4_mM"), ("Cl", "CaCl2_mM"), ("Ca", "CaCl2_mM")]
    for k, (e, s) in enumerate(dose):
        ax = axes_mm(fig, 11 + 35 * k, 5, 27, 33)
        for sess, d in f.groupby("session"):
            mm = d.groupby(s)[f"{e}_d"].mean()
            if len(mm) > 2:
                ax.plot(mm.index, mm.values, color=LIGHT, lw=0.5, zorder=1)
        g = f.groupby(s)[f"{e}_d"]
        med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(med.index, q1, q3, color=ELEMENT[e], alpha=0.2, lw=0, zorder=2)
        ax.plot(med.index, med.values, "o-", color=ELEMENT[e], ms=2.4, mec="white", mew=0.3, zorder=3)
        b, se = ols_fe(f[f"{e}_d"].values, X, f.session.values)
        j = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"].index(s)
        rho = np.median([stats.spearmanr(d[s], d[f"{e}_d"]).correlation for _, d in f.groupby("session") if d[s].nunique() > 2])
        ax.text(0.04, 1.0, e, transform=ax.transAxes, va="top", fontsize=7.5, fontweight="bold", color=ELEMENT[e])
        ax.text(0.04, 0.86, f"β = {b[j]:.2f} ± {se[j]:.2f}\nρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=5.8,
                color="#444444", linespacing=1.3)
        logx(ax, s); ax.set_xlabel(f"{SALT_TEX[s[:-3]]} (mM)")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]) if e == "Mg" else ax.get_ylim()[0])
        if k == 0:
            ax.set_ylabel("Contrast-to-noise ratio, $d$")
        mm_letter(fig, ax, "abcde"[k], dx=-9 if k == 0 else -7)
    # f: co-localisation matrix
    ax = axes_mm(fig, 12, 50, 44, 44)
    M = np.full((7, 7), np.nan)
    for a, b in PAIRS:
        M[b, a] = f[f"x_{EL[a]}-{EL[b]}"].median()
    order = ["Na", "Cl", "Ca", "S", "O", "Mg", "C"]; oi = [EL.index(e) for e in order]
    Mo = np.full((7, 7), np.nan)
    for i in range(7):
        for j in range(i):
            a, b = oi[i], oi[j]
            Mo[i, j] = M[max(a, b), min(a, b)]
    Mo = Mo[1:, :-1]
    cmap = mpl.cm.RdBu_r.copy(); cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(Mo), cmap=cmap, norm=TwoSlopeNorm(0, -0.2, 0.45))
    for i in range(6):
        for j in range(i + 1):
            ax.text(j, i, f"{Mo[i, j]:.2f}".replace("-", "−"), ha="center", va="center", fontsize=5.5,
                    color="white" if abs(Mo[i, j]) > 0.3 else INK)
    for i in range(6):
        for j in range(i + 1):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=0.8))
    ax.set_xticks(range(6)); ax.set_xticklabels(order[:-1]); ax.set_yticks(range(6)); ax.set_yticklabels(order[1:])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cax = axes_mm(fig, 47, 52, 1.8, 20)
    cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.set_label("Excess correlation", fontsize=6)
    mm_letter(fig, ax, "f")
    # g: Ca-S co-localisation against CaCl2 at each NaHCO3 level
    ax = axes_mm(fig, 82, 50, 46, 44)
    for t in range(1, 6):
        d = f[f.table == t].groupby("CaCl2_mM")["x_Ca-S"].agg(["mean", "sem"])
        ax.errorbar(d.index, d["mean"], d["sem"], fmt="o-", color=TABLE[t], ms=2.4, lw=0.9, capsize=0, elinewidth=0.6,
                    mec="white", mew=0.3, label=f"{LEVELS['NaHCO3_mM'][t - 1]:g}")
    logx(ax, "CaCl2_mM"); ax.set_xlabel(f"{SALT_TEX['CaCl2']} (mM)"); ax.set_ylabel("Ca–S excess correlation")
    hd = [mpl.lines.Line2D([], [], color=TABLE[t], marker="o", ms=2.4, mec="white", mew=0.3, lw=0.9) for t in range(1, 6)]
    ax.legend(hd, [f"{v:g}" for v in LEVELS["NaHCO3_mM"]], title=f"{SALT_TEX['NaHCO3']} (mM)", title_fontsize=6, loc="upper left",
              handlelength=1.4, alignment="left")
    mm_letter(fig, ax, "g")
    # h: C-O co-localisation by NaHCO3 level: droplets (jittered) and median with interquartile range
    ax = axes_mm(fig, 146, 50, 32, 44)
    rng = np.random.default_rng(0)
    for i, v in enumerate(LEVELS["NaHCO3_mM"]):
        y = f[f.NaHCO3_mM == v]["x_C-O"].values
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(y)), y, s=1.5, color=TABLE[i + 1], lw=0, alpha=0.8, zorder=1)
        q1, md, q3 = np.percentile(y, [25, 50, 75])
        ax.plot([i - 0.3, i + 0.3], [md, md], color=INK, lw=1.0, zorder=3)
        ax.plot([i, i], [q1, q3], color=INK, lw=0.6, zorder=3)
    ax.axhline(0, color=GREY, lw=0.4, ls=(0, (2, 2)), zorder=0)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{v:g}" for v in LEVELS["NaHCO3_mM"]]); ax.set_xlim(-0.6, 4.6)
    ax.set_xlabel(f"{SALT_TEX['NaHCO3']} (mM)"); ax.set_ylabel("C–O excess correlation")
    mm_letter(fig, ax, "h")
    save(fig, FIG + "/fig2_eds_chemistry")


# ------------------------------------------------------------------ Fig. 3 spatial organisation and grain inventory
def fig3():
    PM = np.load(A + "/out_p/phase_maps.npz", allow_pickle=True)
    names = list(PM["names"]); pi = {p: i for i, p in enumerate(PM["paths"])}
    ph = pd.read_csv(A + "/out_p/phase_droplet.csv"); ph = ph[ph.usable_eds]
    meta = pd.read_csv(ROOT + "/metadata/samples.csv")
    fig = figure(180, 96)
    ex = ["Table_1/Sample_25/1_rep", "Table_2/Sample_13/2_rep", "Table_4/Sample_13/2_rep", "Table_5/Sample_25/1_rep"]
    for k, path in enumerate(ex):
        ax = axes_mm(fig, 4 + 36.5 * k, 6, 35, 26.25)
        row = meta[meta.path == path].iloc[0]
        sem = cv2.resize(cv2.imread(f"{ROOT}/experiment/data/{path}/sem.jpg", 0), (505, 379), interpolation=cv2.INTER_AREA).astype(np.float32)
        sem = (sem - np.percentile(sem, 1)) / (np.percentile(sem, 99) - np.percentile(sem, 1))
        rgb = np.repeat(np.clip(sem, 0, 1)[..., None] * 0.5 + 0.05, 3, -1)
        lab = PM["labels"][pi[path]]
        for j, nm in enumerate(names):
            rgb[lab == j] = mpl.colors.to_rgb(PHASE_COL[nm])
        ax.imshow(rgb, interpolation="nearest"); tidy_image(ax)
        ax.set_title(f"T{row.table} S{row['sample']} R{row.replicate}   {row.NaHCO3_mM:g} · {row.CaCl2_mM:g} · {row.MgSO4_mM:g}",
                     fontsize=6)
        if k == 0:
            mm_letter(fig, ax, "a", dx=-4, dy=-4)
    ax = axes_mm(fig, 150, 7, 28, 25); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for j, nm in enumerate(PHASES):
        y = 0.92 - j * 0.165
        ax.add_patch(Rectangle((0.0, y - 0.05), 0.12, 0.10, color=PHASE_COL[nm]))
        ax.text(0.18, y, nm, va="center", fontsize=6)
    # b: radial profiles, labelled at the rim
    f = features()
    R = np.load(A + "/out_def/radial.npz"); ri = {p: i for i, p in enumerate(R["paths"])}
    rs = np.stack([R["radial_shape"][ri[p]] for p in f.path])
    ax = axes_mm(fig, 12, 40, 44, 48)
    r = (np.arange(12) + 0.5) / 12
    ends = {}
    for e in ["S", "Ca", "Cl", "O", "Na", "C", "Mg"]:
        y = np.nanmedian(rs[:, EL.index(e)], 0)
        ax.plot(r, y, color=ELEMENT[e], lw=1.0)
        ends[e] = y[-1]
    # spread end labels so they do not overlap
    lab_y = dict(sorted(ends.items(), key=lambda kv: kv[1]))
    ys, prev = {}, -9
    for e, v in lab_y.items():
        ys[e] = max(v, prev + 0.13); prev = ys[e]
    for e in ends:
        ax.text(1.02, ys[e], e, color=ELEMENT[e], fontsize=6, va="center", ha="left", fontweight="bold")
    ax.axhline(1, color=GREY, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(0, 1); ax.set_xlabel("Normalised radius"); ax.set_ylabel("Element signal / droplet mean")
    mm_letter(fig, ax, "b")
    # c: grain position of each phase with the predicted precipitation onset in the tick label
    onset = pd.read_csv(A + "/out_t/onset.csv")
    cf = {"gypsum": onset.Gypsum_onset_cf.median(), "Ca carbonate": onset.Calcite_onset_cf.median(),
          "Mg sulfate": onset[["Epsomite_onset_cf", "Kieserite_onset_cf", "Bloedite_onset_cf"]].min(1).median(),
          "halite": onset.Halite_onset_cf.median(), "chloride brine": np.nan, "Na/Mg carbonate": onset.Nesquehonite_onset_cf.median()}
    ax = axes_mm(fig, 92, 40, 34, 48)
    pos = np.arange(len(PHASES))[::-1]
    for y0, nm in zip(pos, PHASES):
        v = ph[f"r_{nm}"].dropna().values
        q = np.percentile(v, [10, 25, 50, 75, 90])
        ax.plot([q[0], q[4]], [y0, y0], color=INK, lw=0.6)
        ax.add_patch(Rectangle((q[1], y0 - 0.28), q[3] - q[1], 0.56, fc=PHASE_COL[nm], ec=INK, lw=0.4))
        ax.plot([q[2], q[2]], [y0 - 0.28, y0 + 0.28], color=INK, lw=0.9)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{nm}\n" + ("no onset" if np.isnan(cf[nm]) else f"onset CF {cf[nm]:,.0f}") for nm in PHASES], fontsize=6,
                       linespacing=1.1)
    ax.set_ylim(-0.6, len(PHASES) - 0.4); ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("Grain position (normalised radius)")
    mm_letter(fig, ax, "c", dx=-24)
    # d: gypsum area over the CaCl2-NaHCO3 plane
    ax = axes_mm(fig, 135, 40, 30, 48)
    grid = ph.groupby(["NaHCO3_mM", "CaCl2_mM"])["area_gypsum"].mean().unstack() * 100
    cm = LinearSegmentedColormap.from_list("gyp", ["#FFFFFF", "#F6C7A8", PHASE_COL["gypsum"], "#7A2B0C"])
    im = ax.imshow(grid.values, origin="lower", cmap=cm, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{v:g}" for v in grid.columns])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"{v:g}" for v in grid.index])
    ax.set_xlabel(f"{SALT_TEX['CaCl2']} (mM)"); ax.set_ylabel(f"{SALT_TEX['NaHCO3']} (mM)")
    for i in range(5):
        for j in range(5):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5, color="white" if v > 4 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cax = axes_mm(fig, 167, 40, 1.8, 48)
    cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.set_label("Gypsum grain area (%)", fontsize=6, labelpad=3)
    mm_letter(fig, ax, "d")
    save(fig, FIG + "/fig3_spatial_phases")


# ------------------------------------------------------------------ Fig. 4 thermodynamic model vs observation
def fig4():
    paths = pd.read_csv(A + "/out_t/evap_paths.csv")
    cmp_ = pd.read_csv(A + "/out_t/compare.csv", index_col=0)
    meta = pd.read_csv(ROOT + "/metadata/samples.csv").drop_duplicates(["table", "sample"])
    meta["recipe"] = "T" + meta.table.astype(str) + "_S" + meta["sample"].astype(str)
    tot = meta.set_index("recipe")[["Ca_mM", "Mg_mM", "Na_mM", "CaCl2_mM", "MgSO4_mM", "NaHCO3_mM"]]
    fig = figure(180, 104)
    ax = axes_mm(fig, 12, 5, 84, 40)
    rec = "T3_S13"; d = paths[paths.recipe == rec]; t = tot.loc[rec]
    cf = 1 / d.mass_H2O.values; aw = 10 ** d.la_H2O.values
    col = lambda m: d[m].values if m in d else np.zeros(len(d))
    curves = [("calcite (Ca)", col("Calcite") / (t.Ca_mM / 1e3), PHASE_COL["Ca carbonate"]),
              ("gypsum (Ca)", col("Gypsum") / (t.Ca_mM / 1e3), PHASE_COL["gypsum"]),
              ("Mg sulfates (Mg)", (col("Epsomite") + col("Hexahydrite") + col("Kieserite") + col("Bloedite")) / (t.Mg_mM / 1e3), PHASE_COL["Mg sulfate"]),
              ("nesquehonite (Mg)", col("Nesquehonite") / (t.Mg_mM / 1e3), PHASE_COL["Na/Mg carbonate"]),
              ("halite (Na)", col("Halite") / (t.Na_mM / 1e3), PHASE_COL["halite"])]
    for lab, y, c_ in curves:
        ax.plot(cf, np.clip(y, 0, 1.05), color=c_, lw=1.1, label=lab)
    i_dry = np.nonzero(aw >= 0.475)[0][-1]
    ax.axvspan(cf[i_dry], cf.max() * 1.3, color="#F0F0F0", lw=0, zorder=0)
    ax.text(cf[i_dry] * 1.12, 0.5, "beyond the\ndrying endpoint", fontsize=6, color="#666666", va="center")
    ax.set_xscale("log"); ax.set_xlim(1, cf.max() * 1.3); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Concentration factor"); ax.set_ylabel("Fraction precipitated")
    ax.legend(loc="upper left")
    mm_letter(fig, ax, "a")
    ax = axes_mm(fig, 124, 9, 52, 36)
    sub = cmp_[(cmp_.CaCl2_mM == 1.0) & (cmp_.MgSO4_mM == 0.75)].sort_values("NaHCO3_mM")
    x = np.arange(len(sub))
    cal, gyp = sub.f_Ca_calcite.clip(0, 1).values, sub.f_Ca_gypsum.clip(0, 1).values
    ax.bar(x, cal, 0.62, color=PHASE_COL["Ca carbonate"], label="calcite")
    ax.bar(x, gyp, 0.62, bottom=cal, color=PHASE_COL["gypsum"], label="gypsum")
    ax.bar(x, np.clip(1 - cal - gyp, 0, 1), 0.62, bottom=cal + gyp, color="#E3E3E3", label="in brine")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in sub.NaHCO3_mM]); ax.set_xlabel(f"{SALT_TEX['NaHCO3']} (mM)")
    ax.set_ylabel("Fraction of Ca"); ax.set_ylim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=0.9)
    mm_letter(fig, ax, "b", dy=-7)
    ph = pd.read_csv(A + "/out_p/phase_droplet.csv"); ph = ph[ph.usable_eds]
    ph["recipe"] = "T" + ph.table.astype(str) + "_S" + ph["sample"].astype(str)
    cmp_ = cmp_.join(ph.groupby("recipe")[["area_gypsum", "area_Mg sulfate", "area_halite"]].mean() * 100, how="inner")
    tests = [("Gypsum_mmol", "area_gypsum", "Predicted gypsum (mmol kg$^{-1}$)", "Gypsum grain area (%)"),
             ("MgSO4_solid_mmol", "area_Mg sulfate", "Predicted Mg sulfates (mmol kg$^{-1}$)", "Mg-sulfate grain area (%)"),
             ("Halite_mmol", "area_halite", "Predicted halite (mmol kg$^{-1}$)", "Halite grain area (%)")]
    for k, (p, o, xl, yl) in enumerate(tests):
        ax = axes_mm(fig, 12 + 60 * k, 59, 44, 36)
        for tb in range(1, 6):
            dd = cmp_[cmp_.table == tb]
            ax.scatter(dd[p], dd[o], s=7, color=TABLE[tb], ec="white", lw=0.25, label=f"{LEVELS['NaHCO3_mM'][tb - 1]:g}", zorder=3)
        within = [stats.spearmanr(g_[p], g_[o]).correlation for _, g_ in cmp_.groupby("table")]
        ax.text(0.03, 0.98, f"ρ = {stats.spearmanr(cmp_[p], cmp_[o]).correlation:.2f}\nρ within NaHCO$_3$ level = {np.median(within):.2f}",
                transform=ax.transAxes, va="top", fontsize=5.8, color="#444444", linespacing=1.3)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.22 * (hi - lo))
        if k == 2:
            ax.legend(title=f"{SALT_TEX['NaHCO3']} (mM)", title_fontsize=6, loc="upper right", handletextpad=0.1,
                      bbox_to_anchor=(1.02, 1.0))
        mm_letter(fig, ax, "cde"[k])
    save(fig, FIG + "/fig4_thermodynamics")


# ------------------------------------------------------------------ Fig. 5 EDS limitations: stretching and Al substrate
def fig5():
    pm = pd.read_csv(A + "/out_c/per_map.csv"); pm = pm[pm.usable_eds]
    f = features()
    R = np.load(A + "/out_def/radial.npz"); ri = {p: i for i, p in enumerate(R["paths"])}
    rad = np.stack([R["radial"][ri[p]] for p in f.path])
    fig = figure(180, 52)
    ax = axes_mm(fig, 12, 4, 38, 38)
    off = {"C": (-3, 3, "right"), "Cl": (0, 4, "center"), "Mg": (4, 1, "left"), "Ca": (-4, 0, "right"),
           "S": (1, 4, "center"), "O": (0, -7, "center"), "Na": (4, -2, "left")}
    for e in EL:
        g = pm[pm.element == e]
        cv_top, cv_mean = g.p999.std() / g.p999.mean(), g["mean"].std() / g["mean"].mean()
        ax.scatter(cv_mean, cv_top, s=14, color=ELEMENT[e], ec="white", lw=0.4, zorder=3)
        dx, dy, ha = off[e]
        ax.annotate(e, (cv_mean, cv_top), xytext=(dx, dy), textcoords="offset points", fontsize=6, ha=ha, color=ELEMENT[e],
                    fontweight="bold")
    ax.plot([0, 0.42], [0, 0.42], ls=(0, (2, 2)), color=GREY, lw=0.5)
    ax.text(0.30, 0.33, "1 : 1", rotation=45, fontsize=5.5, color=GREY, ha="center", va="center")
    ax.set_xlim(0, 0.42); ax.set_ylim(0, 0.42)
    ax.set_xlabel("CV of mean intensity"); ax.set_ylabel("CV of 99.9th-percentile intensity")
    mm_letter(fig, ax, "a")
    ax = axes_mm(fig, 72, 4, 36, 38)
    r = (np.arange(12) + 0.5) / 12
    k = EL.index("Mg")
    greens = LinearSegmentedColormap.from_list("mg", ["#E6C9EA", ELEMENT["Mg"], "#4B1350"])(np.linspace(0, 1, 5))
    for c_, lvl in zip(greens, LEVELS["MgSO4_mM"]):
        y = np.nanmedian(rad[f.MgSO4_mM.values == lvl, k], 0) * 100
        ax.plot(r, y, color=c_, lw=1.0, label=f"{lvl:g}")
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlim(0, 1); ax.set_xlabel("Normalised radius"); ax.set_ylabel("Mg − substrate (% of full scale)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], title=f"{SALT_TEX['MgSO4']} (mM)", title_fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.02))
    mm_letter(fig, ax, "b")
    ax = axes_mm(fig, 138, 4, 40, 38)
    for e in ["Mg", "Na", "S", "Ca", "Cl", "O"]:
        kk = EL.index(e)
        rim = np.nanmean(rad[:, kk, -2:], 1)
        fr = [(rim[f.MgSO4_mM.values == l] < 0).mean() * 100 for l in LEVELS["MgSO4_mM"]]
        ax.plot(LEVELS["MgSO4_mM"], fr, "o-", color=ELEMENT[e], ms=2.2, mec="white", mew=0.3, lw=1.1 if e == "Mg" else 0.7,
                label=e, zorder=3 if e == "Mg" else 2)
    logx(ax, "MgSO4_mM"); ax.set_xlabel(f"{SALT_TEX['MgSO4']} (mM)"); ax.set_ylabel("Rim below substrate (% of droplets)")
    ax.legend(ncol=2, loc="upper right")
    mm_letter(fig, ax, "c")
    save(fig, FIG + "/fig5_eds_limitations")


# ------------------------------------------------------------------ Fig. 6 acquisition structure and SEM detector
def fig6():
    vd = pd.read_csv(A + "/out_b/variance_decomposition.csv")
    vc = pd.read_csv(A + "/out_def/variance_components.csv", index_col=0)
    g = pd.read_csv(A + "/out_g/derived.csv")
    f = features(); g = g[g.path.isin(f.path)]
    G = json.load(open(A + "/out_g/summary.json"))["G3"]
    fig = figure(180, 108)
    ax = axes_mm(fig, 12, 5, 40, 40)
    fam = [("EDS in-droplet chemistry", "#1F4E79", "EDS, in droplet"), ("EDS relative (rim ratio, d)", "#6BAED6", "EDS, relative"),
           ("EDS background (outside droplet)", "#BDBDBD", "EDS, background"),
           ("SEM in-droplet, substrate-normalised", "#C8553D", "SEM, normalised"), ("SEM style (raw, whole image)", "#F2A07B", "SEM, raw"),
           ("Optical global", "#3CB54A", "Optical"), ("Geometry", "#F2B134", "Geometry")]
    for key, c_, lab in fam:
        d = vd[vd.family == key]
        ax.scatter(d.unique_chem, d.unique_session, s=8, color=c_, ec="white", lw=0.3, label=lab, zorder=3)
    ax.plot([0, 0.6], [0, 0.6], ls=(0, (2, 2)), color=GREY, lw=0.5)
    ax.set_xlim(-0.01, 0.6); ax.set_ylim(-0.01, 0.6)
    ax.set_xlabel("Variance unique to chemistry"); ax.set_ylabel("Variance unique to session")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.1, labelspacing=0.45)
    mm_letter(fig, ax, "a")
    ax = axes_mm(fig, 118, 9, 58, 36)
    feats = [("S_d", "S, $d$"), ("O_d", "O, $d$"), ("Mg_d", "Mg, $d$"), ("Ca_d", "Ca, $d$"), ("Cl_d", "Cl, $d$"), ("Na_d", "Na, $d$"),
             ("S_rbar", "S, radius"), ("Mg_rim_index", "Mg, rim index"), ("sem_hf_in", "SEM texture"), ("sem_z_in_mean", "SEM mean grey")]
    feats = [x for x in feats if x[0] in vc.index]
    y = np.arange(len(feats))[::-1]; left = np.zeros(len(feats))
    for comp, c_, lab in [("recipe_within_session", "#1F4E79", "recipe"), ("session", "#F2B134", "session"), ("replicate", "#D9D9D9", "replicate")]:
        v = vc.loc[[x[0] for x in feats], comp].values
        ax.barh(y, v, left=left, color=c_, height=0.72, label=lab); left += v
    ax.set_yticks(y); ax.set_yticklabels([x[1] for x in feats]); ax.set_xlim(0, 1); ax.set_xlabel("Share of variance")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=0.9)
    mm_letter(fig, ax, "b", dx=-26, dy=-7)
    ax = axes_mm(fig, 12, 62, 76, 38)
    order = ["C", "O", "Na", "Mg", "S", "Cl", "Ca"]; Z = [6, 8, 11, 12, 16, 17, 20]
    for i, mode in enumerate(["BEC", "SEI"]):
        d = g[g.sem_signal == mode]
        v = [d[f"semr_{e}"].sub(d[f"semr_{e}_ctrl"]) for e in order]
        med = [x.median() for x in v]; q = [x.quantile([.25, .75]).values for x in v]
        xs = np.arange(7) + (i - 0.5) * 0.34
        ax.bar(xs, med, 0.32, color=DETECTOR[mode], label=f"{mode}, median $R^2$ = {d.sem_on_eds_R2.median():.2f}")
        ax.vlines(xs, [a for a, b in q], [b for a, b in q], color=INK, lw=0.5)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(range(7)); ax.set_xticklabels([f"{e}\n{z}" for e, z in zip(order, Z)], linespacing=1.1)
    ax.set_xlabel("Element and atomic number"); ax.set_ylabel("SEM grey–EDS excess correlation")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    mm_letter(fig, ax, "c", dy=-7)
    ax = axes_mm(fig, 124, 62, 52, 38)
    names = {"sem_z_in_mean": "Mean grey", "sem_z_in_sd": "Grey s.d.", "sem_hf_in": "Texture", "sem_hf_rim": "Rim texture",
             "sem_bright_frac": "Bright fraction", "sem_dark_frac": "Dark fraction", "sem_rim_width": "Rim width"}
    keys = list(names); yy = np.arange(len(keys))[::-1]
    sets = [("BEC − SEI", [G["mode_effect_T2T3"][k] for k in keys], "#7F7F7F", 0.24),
            ("NaHCO$_3$ in BEC", [G["NaHCO3_within_mode_sessionFE"][k]["BEC"] for k in keys], DETECTOR["BEC"], 0.0),
            ("NaHCO$_3$ in SEI", [G["NaHCO3_within_mode_sessionFE"][k]["SEI"] for k in keys], DETECTOR["SEI"], -0.24)]
    for lab, vals, c_, off in sets:
        b = np.array([v[0] for v in vals]); se = np.array([v[1] for v in vals])
        ax.errorbar(b, yy + off, xerr=1.96 * se, fmt="o", color=c_, ms=2.4, mec="white", mew=0.3, elinewidth=0.7, capsize=0, label=lab)
    ax.axvline(0, color=INK, lw=0.5); ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in keys])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Standardised effect (95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.2, handlelength=1.0)
    mm_letter(fig, ax, "d", dx=-22, dy=-7)
    save(fig, FIG + "/fig6_acquisition")


# ------------------------------------------------------------------ Fig. 7 optical modality
def fig7():
    g = pd.read_csv(A + "/out_g/derived.csv"); f = features(); g = g[g.path.isin(f.path) & (g.opt_mask_method != "failed")]
    GS = json.load(open(A + "/out_g/summary.json"))
    fig = figure(180, 50)
    ax = axes_mm(fig, 12, 4, 40, 36)
    for w, d in g.groupby("opt_w"):
        lab = "2592 × 1944 px" if w == 2592 else "1280 × 1024 px"
        ax.scatter(d.eds_diam_px_half * 2, d.opt_diam_px, s=3, alpha=0.75, color="#1F4E79" if w == 2592 else "#F2B134", lw=0, label=lab)
    d = g[g.opt_w == 2592]; r = np.corrcoef(d.eds_diam_px_half, d.opt_diam_px)[0, 1]
    ax.text(0.04, 0.98, f"$r$ = {r:.2f}", transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlabel("EDS droplet diameter (px)"); ax.set_ylabel("Optical droplet diameter (px)")
    lg = ax.legend(loc="lower right", handletextpad=0.2)
    for h in lg.legend_handles:
        h.set_alpha(1); h.set_sizes([10])
    mm_letter(fig, ax, "a")
    ax = axes_mm(fig, 76, 4, 40, 36)
    eff_opt = GS["G5"]["salt_effects_optical_dateFE"]["opt_diam_mm_rel"]
    e3 = json.load(open(A + "/out_def/summary.json"))["E4_size"]["vs_salts"]["session_FE"]
    keys = [("lNa", "NaHCO3"), ("lCa", "CaCl2"), ("lMg", "MgSO4")]
    for i, (kk, s) in enumerate(keys):
        bo = eff_opt[kk]; be = e3[f"log_{s}_mM"]
        ax.errorbar(be[0], i - 0.13, xerr=1.96 * be[1], fmt="o", color="#1F4E79", ms=2.6, mec="white", mew=0.3, elinewidth=0.7,
                    label="EDS mask" if i == 0 else None)
        ax.errorbar(bo[0], i + 0.13, xerr=1.96 * bo[1], fmt="s", color="#3CB54A", ms=2.6, mec="white", mew=0.3, elinewidth=0.7,
                    label="optical image" if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.5); ax.set_yticks(range(3)); ax.set_yticklabels([SALT_TEX[s] for _, s in keys]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Standardised effect on diameter (95% CI)"); ax.legend(loc="lower right")
    mm_letter(fig, ax, "b", dx=-14)
    ax = axes_mm(fig, 138, 4, 40, 36)
    ok = g.optreg_ncc_ctrl.notna(); bins = np.linspace(0, 0.8, 33)
    ax.hist(g.optreg_ncc_ctrl[ok], bins, color="#BDBDBD", label="other replicate")
    ax.hist(g.optreg_ncc[ok], bins, histtype="step", color="#1F4E79", lw=1.0, label="same droplet")
    ax.set_xlabel("Optical–SEM gradient correlation"); ax.set_ylabel("Droplets"); ax.legend(loc="upper right")
    mm_letter(fig, ax, "c")
    save(fig, FIG + "/fig7_optical")


# ------------------------------------------------------------------ Fig. 8 benchmarks
def fig8():
    lin = pd.read_csv(A + "/out_h/h1_table.csv"); cnn = pd.read_csv(A + "/out_m/summary.csv")
    rows = [("Detector label only", "lin", "Detector mode only"), ("EDS descriptors, ridge", "lin", "EDS (hand-crafted)"),
            ("SEM ResNet-18 features, ridge", "lin", "SEM (ResNet18)"), ("Optical ResNet-18 features, ridge", "lin", "Optical (ResNet18)"),
            ("EDS CNN", "cnn", "eds"), ("EDS + SEM CNN", "cnn", "eds_sem"), ("SEM CNN", "cnn", "sem"), ("Optical CNN", "cnn", "opt")]
    salts = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]
    prots = [("P1", "P1 recipe CV"), ("P2", "P2 leave-one-session-out")]
    M = np.full((len(rows), 6), np.nan)
    for i, (_, src, key) in enumerate(rows):
        for j, s in enumerate(salts):
            for q, (pc, pl) in enumerate(prots):
                v = (lin[(lin.input == key) & (lin.target == s) & (lin.protocol == pl)].R2 if src == "lin"
                     else cnn[(cnn.input == key) & (cnn.target == s) & (cnn.protocol == pc)].R2)
                if len(v):
                    M[i, 2 * j + q] = v.iloc[0]
    fig = figure(180, 80)
    # a: R2 with recipes held out (open) and sessions held out (filled), one column per salt; the connecting line is the
    # loss of performance when the acquisition session is new. Groups: frozen features + ridge, fine-tuned ResNet-18.
    short = ["Detector label only", "EDS descriptors", "SEM features", "Optical features", "EDS", "EDS + SEM", "SEM", "Optical"]
    ypos = np.array([0, 1, 2, 3, 4.9, 5.9, 6.9, 7.9])
    for j, sname in enumerate(salts):
        ax = axes_mm(fig, 31 + 25 * j, 12, 22, 54)
        ax.axvspan(-0.25, 0, color="#F2F2F2", lw=0, zorder=0)
        for i in range(len(rows)):
            p1, p2 = M[i, 2 * j], M[i, 2 * j + 1]
            ax.plot([p2, p1], [ypos[i]] * 2, color=SALT[sname[:-3]], lw=1.0, alpha=0.45, zorder=1, solid_capstyle="butt")
            ax.scatter(p1, ypos[i], s=11, facecolor="white", edgecolor=SALT[sname[:-3]], lw=0.8, zorder=3)
            ax.scatter(p2, ypos[i], s=11, color=SALT[sname[:-3]], ec="white", lw=0.3, zorder=4)
        ax.axhline(3.55, color=LIGHT, lw=0.5)
        ax.set_xlim(-0.25, 1.0); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(8.5, -1.1)
        ax.set_yticks(ypos); ax.set_yticklabels(short if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(SALT_TEX[sname[:-3]], fontsize=6.5, fontweight="bold", pad=2)
        if j == 1:
            ax.set_xlabel("$R^2$")
        if j == 0:
            ax.text(-0.27, -0.75, "Frozen features + ridge", transform=ax.transData, ha="right", va="center", fontsize=6,
                    style="italic", color="#555555")
            ax.text(-0.27, 4.15, "Fine-tuned ResNet-18", transform=ax.transData, ha="right", va="center", fontsize=6,
                    style="italic", color="#555555")
        if j == 2:
            hd = [mpl.lines.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, mew=0.8, ms=3.2),
                  mpl.lines.Line2D([], [], ls="", marker="o", color=INK, ms=3.2)]
            ax.legend(hd, ["recipes held out", "sessions held out"], loc="upper left", bbox_to_anchor=(-1.45, -0.13), ncol=2,
                      handletextpad=0.2, columnspacing=1.0)
    letter(fig, 1, 2, "a")
    ax = axes_mm(fig, 118, 12, 24, 48)
    keys = [("eds", "EDS"), ("eds_sem", "EDS + SEM"), ("sem", "SEM"), ("opt", "Optical")]
    for j, s in enumerate(["CaCl2_mM", "MgSO4_mM"]):
        v = np.array([(lambda x: x.iloc[0] if len(x) else np.nan)(cnn[(cnn.input == k) & (cnn.target == s) & (cnn.protocol == "P3")].R2) for k, _ in keys])
        ys = np.arange(len(keys)) + (j - 0.5) * 0.36
        ax.barh(ys, np.clip(v, -1, 1), 0.34, color=SALT[s[:-3]], label=SALT_TEX[s[:-3]])
    ax.axvline(0, color=INK, lw=0.5); ax.set_xlim(-1, 1)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k[1] for k in keys]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Cross-detector $R^2$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.9)
    letter(fig, 106, 2, "b")
    ax = axes_mm(fig, 152, 12, 26, 48)
    p = pd.read_csv(A + "/out_m/preds_eds_P2.csv")
    rng = np.random.default_rng(1)
    for s in salts:
        r = p.groupby(["table", "sample"])[[f"true_{s}", f"pred_{s}"]].mean()
        ax.scatter(2 ** r[f"true_{s}"] * np.exp(rng.normal(0, 0.04, len(r))), 2 ** r[f"pred_{s}"], s=3, color=SALT[s[:-3]], alpha=0.8,
                   lw=0, label=SALT_TEX[s[:-3]])
    ax.plot([0.2, 2.5], [0.2, 2.5], ls=(0, (2, 2)), color=GREY, lw=0.5)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    tk = [0.25, 0.5, 1, 2]
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(mpl.ticker.FixedLocator(tk)); a_.set_major_formatter(mpl.ticker.FixedFormatter([f"{v:g}" for v in tk]))
        a_.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("True (mM)"); ax.set_ylabel("Predicted, recipe mean (mM)")
    lg = ax.legend(loc="upper left", handletextpad=0.1)
    for h in lg.legend_handles:
        h.set_alpha(1); h.set_sizes([10])
    letter(fig, 144, 2, "c")
    save(fig, FIG + "/fig8_benchmarks")


ALL = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8}
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
        print("done", k, flush=True)
