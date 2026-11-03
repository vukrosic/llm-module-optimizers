"""Draws gaps.png from the per-world numbers and means in RESULTS.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [("Muon's recipe vs Muon on every matrix\n250 steps", [.0186, .0230, .0253, .0225, .0300], .0239),
        ("Muon's recipe vs Muon on every matrix\n1,000 steps", [.0092, .0110, .0036, .0033, .0101], .0074),
        ("Muon's recipe vs standard baseline\n250 steps", [.0141, .0132, .0145, .0185, .0207], .0162),
        ("Muon's recipe vs standard baseline\n1,000 steps", [.0144, .0134, .0137, .0091, .0027], .0107),
        ("Lion vs Muon on query and key\n250 steps", [.0101, .0088, .0068, .0069, .0086], .0083),
        ("Lion vs Muon on query and key\n1,000 steps", [-.0038, -.0041, .0012, .0025, .0048], .0001)]
INK, MUTED, DOT = "#1f2328", "#6a7178", "#2a78b5"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=170)
for i, (name, v, m) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    ax.scatter(v, [y] * len(v), s=46, color=DOT, alpha=.55, linewidths=0, zorder=3)
    ax.plot([m, m], [y - .3, y + .3], color=INK, lw=2.2, solid_capstyle="round", zorder=4)
    ax.text(.0322, y, f"{m:+.4f}", va="center", ha="left", color=INK, fontsize=10.5)
ax.axvline(0, color=MUTED, lw=1); ax.axvline(.005, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(.0053, len(ROWS) - .42, "floor .005", ha="left", va="bottom", color=MUTED, fontsize=9.5)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS][::-1], color=INK, fontsize=10); ax.set_xlim(-.006, .0322); ax.set_ylim(-.6, len(ROWS) - .1)
ax.set_xlabel("Loss gap on held-out web text; dots are the 5 worlds, the bar is their mean, positive means the first arm wins", color=MUTED, fontsize=9.5)
ax.tick_params(length=0, colors=MUTED); ax.grid(axis="x", color="#e6e8eb", lw=.8); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("The query and key gain did not last to 1,000 steps", loc="left", color=INK, fontsize=12.5, pad=14)
fig.tight_layout(); fig.savefig("gaps.png"); print("gaps.png")
