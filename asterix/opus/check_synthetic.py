import numpy as np
w = np.load("weights.npz")
rng = np.random.default_rng(0)
ACTS = ["n", "l", "u", "r", "d"]; DXDY = {"n": (0,0), "l": (-1,0), "u": (0,-1), "r": (1,0), "d": (0,1)}
PEN = {0: 6.0, 9: 6.0, 1: 2.0, 8: 2.0}

def make():
    px, py = rng.integers(0, 10), rng.integers(1, 9)
    ents = {}
    for r in range(1, 9):
        if rng.random() < 0.6:
            ents[r] = (int(rng.integers(0, 10)), bool(rng.random() < .5), bool(rng.random() < .35))
    ents.pop(py, None) if (py in ents and ents[py][0] == px) else None   # not already dead
    s = np.zeros((10, 10, 4), bool); s[py, px, 0] = 1
    for r, (x, right, gold) in ents.items():
        s[r, x, 3 if gold else 1] = 1
        bx = x - 1 if right else x + 1
        if 0 <= bx <= 9: s[r, bx, 2] = 1
    return s, px, py, ents

def ref(px, py, ents):
    def unsafe(x, y):
        if y not in ents: return False
        ex, right, gold = ents[y]
        if gold: return False
        return ex == x or (ex == x - 1 and right) or (ex == x + 1 and not right)
    z = np.zeros(6)
    for i, a in enumerate(ACTS):
        dx, dy = DXDY[a]; tx, ty = min(9, max(0, px+dx)), min(8, max(1, py+dy))
        z[i] = -40 * unsafe(tx, ty) - PEN.get(tx, 0)
    golds = [(r, e[0]) for r, e in ents.items() if e[2]]
    if any(r < py for r, _ in golds): z[2] += 3
    if any(r > py for r, _ in golds): z[4] += 3
    if any(r == py and gx < px for r, gx in golds): z[1] += 4
    if any(r == py and gx > px for r, gx in golds): z[3] += 4
    z[5] = -40
    return z

N = 50000; obs = np.zeros((N, 600), np.float32); Z = np.zeros((N, 6))
for n in range(N):
    s, px, py, ents = make(); obs[n, :400] = s.reshape(-1); Z[n] = ref(px, py, ents)
h1 = np.tanh(obs @ w["W1"].T + w["b1"]); h2 = np.tanh(h1 @ w["W2"].T + w["b2"]); z = h2 @ w["W3"].T + w["b3"]
err = np.abs(z - Z).max(1)
print("states", N, "max |logit - ref|", err.max(), "states off by >0.1:", (err > 0.1).sum())
print("min |pre| L1", np.abs(obs @ w["W1"].T + w["b1"]).min(), "L2", np.abs(h1 @ w["W2"].T + w["b2"]).min())
