"""
Hand-written weights for MinAtar Asterix (no training).

Observation: obs[(y*10+x)*4 + ch], ch: 0 player, 1 enemy, 2 trail, 3 gold; obs[400:] = 0.
Actions: 0 n, 1 l, 2 u, 3 r, 4 d, 5 f.

Policy (reactive, player-centric):
  * Unsafe(r,c): enemy sits on (r,c), or an enemy within 3 cells in row r is moving
    toward c (direction read from the trail channel; an entity at x=0/9 without a
    trail is a fresh spawn moving inward).  Border quirk (accepted): an enemy at
    x=0 counts as unsafe for c=1..3 regardless of direction (same for x=9, c=6..8).
  * T  = player cell is Unsafe, or an enemy (any direction) is horizontally adjacent.
  * UU = cell above the player is Unsafe (or player is in row 1, where 'u' is a no-op).
  * DU = cell below is Unsafe (or row 8).
  * GA/GB: gold anywhere in a row above/below the player. GL/GR: gold anywhere in a
    column left/right of the player.
  * E01/E2/E7/E89: player column in {0,1}, 2, 7, {8,9} (spawn zones, push inward).
Logits:
  n = 4 - 20 T
  u = 12 T - 30 UU + 8 GA + 3 Row8
  d = 12 T - 30 DU + 8 GB + 3 Row1
  l = 8 GL - 8 T + 10 E89 + 5 E7
  r = 8 GR - 8 T + 10 E01 + 5 E2
  f = -50
Layer 1 (256 tanh units): all the per-cell ANDs with the player's position (T, UU,
DU per player cell) plus the row/column gold and edge detectors, each a sharp linear
threshold on the boolean inputs.  Layer 2 (11 units): ORs over the layer-1 groups.
Layer 3: the linear logit table above (units mapped from +-1 to 0/1).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P, E, TR, G = 0, 1, 2, 3
K1 = 8.0   # layer-1 gain (margins are >= 0.5 -> |tanh| >= 0.9993)
K2 = 8.0   # layer-2 gain


def idx(y, x, ch):
    return (y * 10 + x) * 4 + ch


def unsafe_terms(r, c, adj_any=False):
    """Linear score S(r,c) over raw inputs; Unsafe <=> S >= 3.5 (S is an integer).
    Approaching enemy at distance j contributes w_e[j] + w_t[j+1] = 4; receding ones <= 3;
    an enemy on the cell itself gives 4; lone trails (gold) give <= 3."""
    w = {}

    def add(i, v):
        w[i] = w.get(i, 0.0) + v

    add(idx(r, c, E), 4.0)
    we = {1: 3.0, 2: 2.0, 3: 1.0}
    if adj_any:
        we[1] = 4.0            # adjacent enemy of either direction counts
    wt = {2: 1.0, 3: 2.0, 4: 3.0}
    for j in (1, 2, 3):
        cl = c - j
        if cl == 0:
            add(idx(r, 0, E), 4.0)          # no trail possible for a fresh spawn
        elif cl > 0:
            add(idx(r, cl, E), we[j])
            add(idx(r, cl - 1, TR), wt[j + 1])
        cr = c + j
        if cr == 9:
            add(idx(r, 9, E), 4.0)
        elif cr < 9:
            add(idx(r, cr, E), we[j])
            add(idx(r, cr + 1, TR), wt[j + 1])
    return w


def build():
    rows_l1 = []   # list of (weights dict, bias) for layer 1
    names = []

    def unit(wdict, thresh, name):
        rows_l1.append((wdict, -thresh))
        names.append(name)
        return len(rows_l1) - 1

    groups = {k: [] for k in ("T", "UU", "DU", "GA", "GB", "GL", "GR",
                              "E01", "E2", "E7", "E89", "ROW1", "ROW8")}

    # T(x,y): player at (x,y) AND (Unsafe(y,x) or adjacent enemy)
    for y in range(1, 9):
        for x in range(10):
            w = unsafe_terms(y, x, adj_any=True)
            w[idx(y, x, P)] = w.get(idx(y, x, P), 0.0) + 10.0
            groups["T"].append(unit(w, 13.5, f"T{x},{y}"))
    # UU(x,y) for y=2..8 : player at (x,y) AND Unsafe(y-1,x)
    for y in range(2, 9):
        for x in range(10):
            w = unsafe_terms(y - 1, x)
            w[idx(y, x, P)] = w.get(idx(y, x, P), 0.0) + 10.0
            groups["UU"].append(unit(w, 13.5, f"UU{x},{y}"))
    # DU(x,y) for y=1..7
    for y in range(1, 8):
        for x in range(10):
            w = unsafe_terms(y + 1, x)
            w[idx(y, x, P)] = w.get(idx(y, x, P), 0.0) + 10.0
            groups["DU"].append(unit(w, 13.5, f"DU{x},{y}"))
    # row 1 / row 8 (vertical no-op direction)
    w = {idx(1, x, P): 1.0 for x in range(10)}
    u = unit(w, 0.5, "ROW1"); groups["ROW1"].append(u); groups["UU"].append(u)
    w = {idx(8, x, P): 1.0 for x in range(10)}
    u = unit(w, 0.5, "ROW8"); groups["ROW8"].append(u); groups["DU"].append(u)
    # gold above / below
    for y in range(2, 9):
        w = {idx(y, x, P): 10.0 for x in range(10)}
        for r in range(1, y):
            for c in range(10):
                w[idx(r, c, G)] = 1.0
        groups["GA"].append(unit(w, 10.5, f"GA{y}"))
    for y in range(1, 8):
        w = {idx(y, x, P): 10.0 for x in range(10)}
        for r in range(y + 1, 9):
            for c in range(10):
                w[idx(r, c, G)] = 1.0
        groups["GB"].append(unit(w, 10.5, f"GB{y}"))
    # gold left / right
    for x in range(2, 10):
        w = {idx(y, x, P): 10.0 for y in range(1, 9)}
        for r in range(1, 9):
            for c in range(0, x):
                w[idx(r, c, G)] = 1.0
        groups["GL"].append(unit(w, 10.5, f"GL{x}"))
    for x in range(0, 8):
        w = {idx(y, x, P): 10.0 for y in range(1, 9)}
        for r in range(1, 9):
            for c in range(x + 1, 10):
                w[idx(r, c, G)] = 1.0
        groups["GR"].append(unit(w, 10.5, f"GR{x}"))
    # edge columns
    for key, cols in (("E01", (0, 1)), ("E2", (2,)), ("E7", (7,)), ("E89", (8, 9))):
        w = {idx(y, x, P): 1.0 for y in range(1, 9) for x in cols}
        groups[key].append(unit(w, 0.5, key))

    H1 = len(rows_l1)
    assert H1 == 256, H1
    W1 = np.zeros((H1, 600))
    b1 = np.zeros(H1)
    for i, (w, b) in enumerate(rows_l1):
        for j, v in w.items():
            W1[i, j] = K1 * v
        b1[i] = K1 * b

    # layer 2: OR over each group, outputs ~ +-1
    order = ["T", "UU", "DU", "GA", "GB", "GL", "GR", "E01", "E2", "E7", "E89", "ROW1", "ROW8"]
    H2 = len(order)
    W2 = np.zeros((H2, H1))
    b2 = np.zeros(H2)
    for j, key in enumerate(order):
        members = groups[key]
        for i in members:
            W2[j, i] = K2 * 0.5
        b2[j] = K2 * (len(members) / 2.0 - 0.5)

    # layer 3: logits from 0/1-mapped units  (contrib = w*(h+1)/2)
    N, L, U, R, D, F = range(6)
    terms = {
        N: {"T": -20.0},
        U: {"T": 12.0, "UU": -30.0, "GA": 8.0, "ROW8": 3.0},
        D: {"T": 12.0, "DU": -30.0, "GB": 8.0, "ROW1": 3.0},
        L: {"GL": 8.0, "T": -8.0, "E89": 10.0, "E7": 5.0},
        R: {"GR": 8.0, "T": -8.0, "E01": 10.0, "E2": 5.0},
        F: {},
    }
    const = {N: 4.0, L: 0.0, U: 0.0, R: 0.0, D: 0.0, F: -50.0}
    W3 = np.zeros((6, H2))
    b3 = np.zeros(6)
    for a in range(6):
        b3[a] = const[a]
        for key, wv in terms[a].items():
            j = order.index(key)
            W3[a, j] += wv / 2.0
            b3[a] += wv / 2.0
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3), names, order


def forward(wts, obs):
    h1 = np.tanh(obs @ wts["W1"].T + wts["b1"])
    h2 = np.tanh(h1 @ wts["W2"].T + wts["b2"])
    return h2 @ wts["W3"].T + wts["b3"], h1, h2


# ---------------------------------------------------------------------------
# Plain reference implementation of the intended rule (semantic, not weight-based)
# ---------------------------------------------------------------------------
def reference_logits(obs):
    s = obs[:400].reshape(10, 10, 4)
    ys, xs = np.argwhere(s[:, :, P] == 1)[0]
    y, x = int(ys), int(xs)
    enemies = {}  # row -> (col, dir)  dir +1 right, -1 left
    golds = []
    for r in range(1, 9):
        for c in range(10):
            if s[r, c, E] or s[r, c, G]:
                if c >= 1 and s[r, c - 1, TR]:
                    d = +1
                elif c <= 8 and s[r, c + 1, TR]:
                    d = -1
                else:
                    d = +1 if c == 0 else -1
                if s[r, c, E]:
                    enemies[r] = (c, d)
                else:
                    golds.append((r, c))

    def unsafe(r, c):
        if r not in enemies:
            return False
        p, d = enemies[r]
        if p == c:
            return True
        if p < c and c - p <= 3 and d == +1:
            return True
        if p > c and p - c <= 3 and d == -1:
            return True
        if p == 0 and 1 <= c <= 3:      # accepted border quirk
            return True
        if p == 9 and 6 <= c <= 8:
            return True
        return False

    adj = y in enemies and abs(enemies[y][0] - x) == 1
    T = unsafe(y, x) or adj
    UU = (y == 1) or unsafe(y - 1, x)
    DU = (y == 8) or unsafe(y + 1, x)
    GA = any(r < y for r, c in golds)
    GB = any(r > y for r, c in golds)
    GL = x >= 2 and any(c < x for r, c in golds)   # no GL unit at x=1 (gold in col 0 not chased)
    GR = x <= 7 and any(c > x for r, c in golds)   # no GR unit at x=8
    E01, E2, E7, E89 = x in (0, 1), x == 2, x == 7, x in (8, 9)
    R1, R8 = y == 1, y == 8
    z = np.zeros(6)
    z[0] = 4 - 20 * T
    z[1] = 8 * GL - 8 * T + 10 * E89 + 5 * E7
    z[2] = 12 * T - 30 * UU + 8 * GA + 3 * R8
    z[3] = 8 * GR - 8 * T + 10 * E01 + 5 * E2
    z[4] = 12 * T - 30 * DU + 8 * GB + 3 * R1
    z[5] = -50
    return z


def synthetic_obs(rng, density=0.6):
    """Random state consistent with the game's state() layout (my own construction)."""
    s = np.zeros((10, 10, 4))
    y = rng.integers(1, 9)
    x = rng.integers(0, 10)
    s[y, x, P] = 1
    for r in range(1, 9):
        if rng.random() < density:
            c = rng.integers(0, 10)
            if r == y and c == x:
                continue   # cannot coexist with the player
            right = rng.random() < 0.5
            gold = rng.random() < 1 / 3
            s[r, c, G if gold else E] = 1
            back = c - 1 if right else c + 1
            if 0 <= back <= 9:
                s[r, back, TR] = 1
    obs = np.zeros(600)
    obs[:400] = s.reshape(-1)
    return obs


def main():
    wts, names, order = build()
    np.savez(os.path.join(HERE, "weights.npz"), **wts)
    print("saved", {k: v.shape for k, v in wts.items()})

    rng = np.random.default_rng(0)
    worst = 0.0
    n_mismatch = 0
    N = 20000
    for i in range(N):
        obs = synthetic_obs(rng, density=rng.choice([0.3, 0.6, 1.0]))
        z, h1, h2 = forward(wts, obs)
        zr = reference_logits(obs)
        err = np.max(np.abs(z - zr))
        worst = max(worst, err)
        if err > 0.2:
            n_mismatch += 1
            if n_mismatch <= 5:
                print("MISMATCH", z.round(2), zr.round(2))
    print(f"synthetic checks: {N} states, max |logit error| = {worst:.4f}, mismatches = {n_mismatch}")

    # a few hand-made scenarios, printed for eyeballing
    def mk(player, ents):
        s = np.zeros((10, 10, 4))
        s[player[1], player[0], P] = 1
        for (r, c, d, gold) in ents:
            s[r, c, G if gold else E] = 1
            back = c - 1 if d > 0 else c + 1
            if 0 <= back <= 9:
                s[r, back, TR] = 1
        o = np.zeros(600); o[:400] = s.reshape(-1); return o

    scen = {
        "calm center": mk((5, 5), []),
        "enemy 2 left approaching, rows clear": mk((5, 5), [(5, 3, +1, False)]),
        "enemy 2 left receding": mk((5, 5), [(5, 3, -1, False)]),
        "enemy adjacent receding": mk((5, 5), [(5, 4, -1, False)]),
        "danger, up unsafe (enemy above approaching)": mk((5, 5), [(5, 7, -1, False), (4, 4, +1, False)]),
        "trapped": mk((5, 5), [(5, 7, -1, False), (4, 4, +1, False), (6, 6, -1, False)]),
        "gold above": mk((5, 5), [(2, 8, -1, True)]),
        "gold left same row": mk((5, 5), [(5, 1, +1, True)]),
        "edge col 0 calm": mk((0, 5), []),
        "row 1 danger": mk((5, 1), [(1, 3, +1, False)]),
        "spawn at 0, player col 3": mk((3, 5), [(5, 0, +1, False)]),
        "spawn at 0, player col 4": mk((4, 5), [(5, 0, +1, False)]),
    }
    for k, o in scen.items():
        z, _, _ = forward(wts, o)
        p = np.exp(z - z.max()); p /= p.sum()
        print(f"{k:48s} z={np.round(z, 1)}  p(n,l,u,r,d)={np.round(p[:5], 3)}")


if __name__ == "__main__":
    main()
