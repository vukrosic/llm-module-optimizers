"""Draws attention.png from the 600-step table in EXPERIMENTS.md, round 11: forgetting on one axis, relearning on the other."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

ROWS = [("Muon", .493, 2.153, "b", (8, 0)), ("Per-head Muon", .231, 1.980, "b", (8, 0)), ("Adam", .265, 1.679, "o", (8, 3)), ("Lion", .341, 1.548, "o", (8, 0)), ("NormSGD", .595, 1.191, "o", (8, 0)), ("Adam, tangent per head", 3.384, 1.047, "n", (-8, 0)), ("Muon, tangent per head", .261, 1.335, "n", (8, 0)), ("Muon on a sphere per head", .161, 1.596, "n", (-8, 0)), ("Muon with gradient history", 1.421, .677, "n", (8, 0))]
INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#6a7178", "#2a78b5", "#c8561b", "#9aa1a8"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=170)
for name, x, y, kind, off in ROWS:
    ax.scatter([x], [y], s=70, color={"b": BLUE, "o": GREY, "n": ORANGE}[kind], linewidths=1.5, edgecolors="white", zorder=3)
    ax.annotate(name, (x, y), xytext=off, textcoords="offset points", color=INK, fontsize=10, ha="right" if off[0] < 0 else "left", va="center")
ax.set_xscale("log"); ax.set_xlim(.04, 6); ax.set_ylim(.4, 2.4)
ax.xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2, 5])); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("Forgetting: loss on old cues after training on new ones (log scale, lower is better)", color=MUTED, fontsize=10); ax.set_ylabel("Relearning: loss while cues change meaning\n(lower is better)", color=MUTED, fontsize=10)
ax.tick_params(length=0, colors=MUTED); ax.grid(color="#e6e8eb", lw=.8); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("The two attention rules that relearn fastest forget the most, at 600 steps", loc="left", color=INK, fontsize=12.5, pad=14)
fig.text(.01, .01, "Blue: main baselines. Grey: other baselines. Orange: rules written here. Mean of 5 worlds, lrs picked for memory first.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig("attention.png"); print("attention.png")
