"""Hand-written Asterix policy (no training, no game runs).

Rule: never take an action whose target cell can be hit this step; among safe
actions, move toward gold (same row: horizontally; other rows: vertically) and
stay away from the edge columns where entities spawn.

Cell c is unsafe if an enemy is on it (hit before enemies move) or an enemy is
next to it in the row and heading toward it (hit if enemies move this step;
the move timer is not observable). With one entity per row, an enemy at c-1
or c+1 heads toward c exactly when c holds no trail. So
    s(c) = 2E(c) + E(c-1) + E(c+1) - T(c),  unsafe(c) <=> s(c) >= 1.
"""
import numpy as np

P, E, T, G = 0, 1, 2, 3
def idx(y, x, c): return (y * 10 + x) * 4 + c
BETA = 20.0
ACTS = ["n", "l", "u", "r", "d"]          # indices 0..4; 5 = 'f' (no effect)
DXDY = {"n": (0, 0), "l": (-1, 0), "u": (0, -1), "r": (1, 0), "d": (0, 1)}

rows1, cols1 = [], []                      # layer 1 rows of W1, biases
def l1(w, b):
    rows1.append(w); cols1.append(b); return len(rows1) - 1

def py_vec(r):                             # sum of player inputs in row r
    w = np.zeros(600)
    for x in range(10): w[idx(r, x, P)] = 1
    return w

# --- L1: R[(r, r2, x)] = player in row r AND cell (x, r2) unsafe
R = {}
pairs = [(r, r) for r in range(1, 9)] + [(r, r - 1) for r in range(2, 9)] + [(r, r + 1) for r in range(1, 8)]
for (r, r2) in pairs:
    for x in range(10):
        w = np.zeros(600)
        w[idx(r2, x, E)] += 2
        if x > 0: w[idx(r2, x - 1, E)] += 1
        if x < 9: w[idx(r2, x + 1, E)] += 1
        w[idx(r2, x, T)] -= 1
        w += 4 * py_vec(r)
        R[(r, r2, x)] = l1(BETA * w, BETA * (-0.5 - 4))

# --- L1: player column one-hot
PX = []
for x in range(10):
    w = np.zeros(600)
    for y in range(10): w[idx(y, x, P)] = 1
    PX.append(l1(BETA * w, BETA * -0.5))

# --- L1: any gold in a row above / below the player (exponential weights)
def gold_dir(above):
    w = np.zeros(600)
    for r in range(1, 9):
        a = 2.0 ** -r if above else 2.0 ** -(9 - r)
        for x in range(10): w[idx(r, x, G)] += a
    for k in range(1, 9):
        b = 2.0 ** -(k - 1) - 2.0 ** -9 if above else 2.0 ** -(9 - k - 1) - 2.0 ** -9
        w -= b * py_vec(k)
    g = BETA * 2 ** 9
    return l1(g * w, 0.0)
GA, GB = gold_dir(True), gold_dir(False)

# --- L1: gold in the player's row, left / right of the player
pxw = np.zeros(600)
for y in range(10):
    for x in range(10): pxw[idx(y, x, P)] = x
GL, GR = [], []
for r in range(1, 9):
    w = np.zeros(600)
    for x in range(10): w[idx(r, x, G)] = 10 - x
    w += pxw + 12 * py_vec(r)
    GL.append(l1(BETA * w, BETA * (-10.5 - 12)))
    w = np.zeros(600)
    for x in range(10): w[idx(r, x, G)] = x + 1
    w += -pxw + 12 * py_vec(r)
    GR.append(l1(BETA * w, BETA * (-1.5 - 12)))

H1 = len(rows1)
W1 = np.array(rows1); b1 = np.array(cols1)

# --- L2. Inputs are tanh units; q = (h+1)/2 is the 0/1 value.
rows2, b2l = [], []
def l2(terms, const):
    """pre = BETA * (sum coef*q(h) + const)"""
    w = np.zeros(H1); b = const
    for j, c in terms:
        w[j] += c / 2; b += c / 2
    rows2.append(BETA * w); b2l.append(BETA * b); return len(rows2) - 1

SAFE = {a: [] for a in ACTS}               # L2 units: action a hits unsafe target tx
for a in ACTS:
    dx, dy = DXDY[a]
    for tx in range(10):
        src = [px for px in range(10) if min(9, max(0, px + dx)) == tx]
        if not src: continue
        rt = []
        for r in range(1, 9):
            r2 = min(8, max(1, r + dy))
            rt.append((R[(r, r2, tx)], 1.0))
        SAFE[a].append(l2(rt + [(PX[px], 1.0) for px in src], -1.5))
UGA = l2([(GA, 1.0)], -0.5)
UGB = l2([(GB, 1.0)], -0.5)
UGL = l2([(j, 1.0) for j in GL], -0.5)
UGR = l2([(j, 1.0) for j in GR], -0.5)
UPX = [l2([(PX[x], 1.0)], -0.5) for x in range(10)]
H2 = len(rows2)
W2 = np.array(rows2); b2 = np.array(b2l)

# --- output logits
K, A_V, A_H = 40.0, 3.0, 4.0
PEN = {0: 6.0, 9: 6.0, 1: 2.0, 8: 2.0}
W3 = np.zeros((6, H2)); b3 = np.zeros(6)
def add(a, j, c):                          # logit_a += c * q(h2_j)
    W3[a, j] += c / 2; b3[a] += c / 2
for ai, a in enumerate(ACTS):
    for j in SAFE[a]: add(ai, j, -K)
    dx, _ = DXDY[a]
    for px in range(10):
        tx = min(9, max(0, px + dx))
        if tx in PEN: add(ai, UPX[px], -PEN[tx])
add(ACTS.index("u"), UGA, A_V); add(ACTS.index("d"), UGB, A_V)
add(ACTS.index("l"), UGL, A_H); add(ACTS.index("r"), UGR, A_H)
b3[5] = -40.0                              # 'f' does nothing in Asterix; noop covers it

if __name__ == "__main__":
    np.savez("weights.npz", W1=W1.astype(np.float32), b1=b1.astype(np.float32),
             W2=W2.astype(np.float32), b2=b2.astype(np.float32),
             W3=W3.astype(np.float32), b3=b3.astype(np.float32))
    print("H1", H1, "H2", H2)
