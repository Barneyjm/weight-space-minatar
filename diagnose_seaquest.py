"""Post-hoc failure analysis (after one-shot scoring). Usage: diag weights.npz out.txt"""
import sys, collections, numpy as np
from minatar import Environment
w = np.load(sys.argv[1]); out = open(sys.argv[2], "w")
rng = np.random.default_rng(777); ACTS = "nlurdf"
def logits(s):
    h1 = np.tanh(s @ w["W1"].T + w["b1"]); h2 = np.tanh(h1 @ w["W2"].T + w["b2"]); return h2 @ w["W3"].T + w["b3"]
def render(st):
    g = [["." for _ in range(10)] for _ in range(10)]
    marks = [(9, "d"), (5, "F"), (6, "S"), (4, "*"), (2, "-"), (1, "b"), (0, "P")]
    for c, ch in marks:
        for y, x in zip(*np.nonzero(st[:, :, c])): g[y][x] = ch
    for y, x in zip(*np.nonzero(st[:, :, 3])):
        if g[y][x] == ".": g[y][x] = "~"
    ox = int(st[9, :, 7].sum()); dv = int(st[9, :, 8].sum())
    return "\n".join("".join(r) for r in g[:9]) + f"\n oxygen bar {ox}/10, divers {dv}"
causes = collections.Counter(); examples = collections.defaultdict(list); lengths = []
for ep in range(300):
    e = Environment("seaquest", sticky_action_prob=0.1); e.seed(10_000 + ep); e.reset()
    hist = []
    for t in range(10_000):
        st = e.state(); s = st.reshape(-1).astype(float); z = logits(s)
        p = np.exp(z - z.max()); p /= p.sum(); a = min(5, int((p.cumsum() < rng.random()).sum()))
        g = e.env; ox, dc, y0 = g.oxygen, g.diver_count, g.sub_y
        _, done = e.act(a); sticky = e.last_action != a
        hist.append((st, z, a, e.last_action))
        if done:
            if ox <= 0 or g.oxygen <= 0: c = "oxygen ran out"
            elif g.sub_y == 0 and dc == 0: c = "surfaced with no divers"
            else: c = "hit by enemy/bullet"
            c += " (sticky repeat)" if sticky else ""
            causes[c] += 1; lengths.append(t + 1)
            if len(examples[c]) < 4: examples[c].append(hist[-3:])
            break
print(f"300 diagnostic episodes (seeds 10000+, not the scoring seeds). Mean length {np.mean(lengths):.0f}.", file=out)
print("Cause of death:", file=out)
for c, n in causes.most_common(): print(f"  {n:4d}  {c}", file=out)
print("\nLegend: P sub front, b sub back, F fish, S enemy sub, * enemy bullet, - your bullet, d diver, ~ trail", file=out)
for c, exs in examples.items():
    print(f"\n===== {c}: last 3 frames of {len(exs)} episodes =====", file=out)
    for k, frames in enumerate(exs):
        print(f"--- example {k+1}", file=out)
        for st, z, a, done_a in frames:
            print(render(st), file=out)
            print(f" logits n l u r d f = {np.round(z, 1).tolist()}  chosen {ACTS[a]}  executed {ACTS[done_a]}\n", file=out)
