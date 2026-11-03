"""Draws gaps.png from the per-world numbers in RESULTS.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [("Muon recipe vs standard baseline\n250 steps", [.0141, .0132, .0145, .0185, .0207]),
        ("Muon recipe vs standard baseline\n1,000 steps", [.0144, .0134, .0137, .0091, .0027]),
        ("Rule search vs best recipe\n250 steps", [.0180, .0072, .0067, .0056, -.0010]),
        ("Rule search vs lr-only search\n250 steps", [.0080, .0047, .0024, .0041, .0007]),
        ("Rule search vs best recipe\n1,000 steps", [-.0005, .0012, .0001, .0019, -.0013])]
INK, MUTED, DOT = "#1f2328", "#6a7178", "#2a78b5"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
for i, (name, v) in enumerate(ROWS):
    y = len(ROWS) - 1 - i; m = sum(v) / len(v)
    ax.scatter(v, [y] * len(v), s=46, color=DOT, alpha=.55, linewidths=0, zorder=3)
    ax.plot([m, m], [y - .3, y + .3], color=INK, lw=2.2, solid_capstyle="round", zorder=4)
    ax.text(.0222, y, f"{m:+.4f}", va="center", ha="left", color=INK, fontsize=10.5)
ax.axvline(0, color=MUTED, lw=1); ax.axvline(.005, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(.0053, len(ROWS) - .42, "floor .005", ha="left", va="bottom", color=MUTED, fontsize=9.5)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS][::-1], color=INK, fontsize=10); ax.set_xlim(-.003, .0222); ax.set_ylim(-.6, len(ROWS) - .1)
ax.set_xlabel("Loss gap on held-out web text; dots are the 5 worlds, the bar is their mean, positive means the first arm wins", color=MUTED, fontsize=9.5)
ax.tick_params(length=0, colors=MUTED); ax.grid(axis="x", color="#e6e8eb", lw=.8); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Searching update rules per module did not clear the floor against lr tuning", loc="left", color=INK, fontsize=12.5, pad=14)
fig.tight_layout(); fig.savefig("gaps.png"); print("gaps.png")
