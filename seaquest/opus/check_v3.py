import numpy as np
w = np.load("weights_v3.npz"); rng = np.random.default_rng(1)
ACTS = ["n", "l", "u", "r", "d", "f"]; DXDY = {"n": (0,0), "l": (-1,0), "u": (0,-1), "r": (1,0), "d": (0,1)}
cl = lambda v, lo, hi: min(hi, max(lo, v))

def make():
    while True:
        x, y, right = int(rng.integers(10)), int(rng.integers(9)), bool(rng.random() < .5)
        if not ((x == 0 and right) or (x == 9 and not right)): break
    oxy, ndiv = int(rng.integers(0, 11)), int(rng.integers(0, 7))
    enemies, divers, ebul = [], [], set()
    for r in range(1, 9):
        d = bool(rng.random() < .5)
        for ex in set(rng.integers(0, 10, rng.integers(0, 3)).tolist()):
            enemies.append((int(ex), r, d, bool(rng.random() < 1/3)))
        if rng.random() < .25: divers.append((int(rng.integers(10)), r, bool(rng.random() < .5)))
        if rng.random() < .2: ebul.add((int(rng.integers(10)), r))
    enemies = [e for e in enemies if (e[0], e[1]) != (x, y)]; ebul.discard((x, y))
    s = np.zeros((10, 10, 10), bool)
    s[y, x, 0] = 1; s[y, x - 1 if right else x + 1, 1] = 1
    s[9, :oxy, 7] = 1; s[9, 9 - ndiv:9, 8] = 1
    for bx, by in ebul: s[by, bx, 4] = 1
    for ex, ey, d, sub in enemies:
        s[ey, ex, 6 if sub else 5] = 1; b = ex - 1 if d else ex + 1
        if 0 <= b <= 9: s[ey, b, 3] = 1
    for dx_, dy_, d in divers:
        s[dy_, dx_, 9] = 1; b = dx_ - 1 if d else dx_ + 1
        if 0 <= b <= 9: s[dy_, b, 3] = 1
    return s, (x, y, right, oxy, ndiv, enemies, divers, ebul)

def ref(x, y, right, oxy, ndiv, enemies, divers, ebul):
    epos = {(e[0], e[1]): e for e in enemies}
    def unsafe(cx, cy):
        if cy == 0: return False
        if (cx, cy) in epos: return True
        if any((cx + k, cy) in ebul for k in (-1, 0, 1)): return True
        return any(ex == cx - 1 and d or ex == cx + 1 and not d for ex, ey, d, _ in enemies if ey == cy)
    z = np.zeros(6)
    for a, (dx, dy) in DXDY.items():
        z[ACTS.index(a)] -= 40 * unsafe(cl(x + dx, 0, 9), cl(y + dy, 0, 8))
    z[5] += -40 * unsafe(x, y) + 3; z[0] += 2
    row = [e[0] for e in enemies if e[1] == y]
    if right and any(ex > x for ex in row) or (not right) and any(ex < x for ex in row): z[5] += 5
    if any(ex < x for ex in row): z[1] -= 6
    if any(ex > x for ex in row): z[3] -= 6
    if x <= 1: z[1] -= 4
    if x >= 8: z[3] -= 4
    low, has = oxy <= 2, ndiv >= 1
    if y == 1 and not has: z[2] -= 40; z[4] += 5
    if y == 2 and not has: z[2] -= 40
    if y == 1 and has and not low: z[2] -= 4
    if low and has: z[2] += 10
    if any(r < y for _, r, _ in divers): z[2] += 4.5
    if any(r > y for _, r, _ in divers): z[4] += 4.5
    if y == 0: z[4] += 6
    if y >= 3 and unsafe(x, y - 2): z[2] -= 8
    if y <= 6 and unsafe(x, y + 2): z[4] -= 8
    if ndiv == 6: z[2] += 12
    return z

N = 40000; obs = np.zeros((N, 1000)); Z = np.zeros((N, 6)); info = []
for n in range(N):
    s, st = make(); obs[n] = s.reshape(-1); Z[n] = ref(*st); info.append(st)
h1 = np.tanh(obs @ w["W1"].T + w["b1"]); h2 = np.tanh(h1 @ w["W2"].T + w["b2"]); z = h2 @ w["W3"].T + w["b3"]
bad = np.abs(z - Z).max(1) > 0.5
print("states", N, "mismatched", bad.sum(), "max err on matched", np.abs(z - Z)[~bad].max())
print("min |pre| L1", np.abs(obs @ w["W1"].T + w["b1"]).min().round(2), "L2", np.abs(h1 @ w["W2"].T + w["b2"]).min().round(2))
# classify mismatches: are they all trail-ambiguity cases (another entity's trail on the target cell)?
for n in np.flatnonzero(bad)[:5]:
    print("mismatch", n, "net", z[n].round(1), "ref", Z[n].round(1))
# Is every mismatch a diver trail landing on the target cell (the known blind spot)?
def diver_trail_cells(divers):
    return {(dx_ - 1 if d else dx_ + 1, dy_) for dx_, dy_, d in divers}
explained = 0
for n in np.flatnonzero(bad):
    x, y, right, oxy, ndiv, enemies, divers, ebul = info[n]
    tc = diver_trail_cells(divers)
    cells = [(cl(x + dx, 0, 9), cl(y + dy, 0, 8)) for dx, dy in DXDY.values()] + [(x, y - 2), (x, y + 2)]
    explained += any(c in tc for c in cells)
print("mismatches with a diver trail on a target cell:", explained, "of", bad.sum(),
      "| net says safe where ref says unsafe:", int(((z - Z)[bad] > 0.5).any(1).sum()))
