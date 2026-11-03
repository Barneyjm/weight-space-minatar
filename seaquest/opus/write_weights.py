"""Hand-written Seaquest policy (no training, no game runs).

Danger: cell (x, y) is unsafe if an enemy is on it, an enemy next to it in the
row is heading toward it, or an enemy bullet is on it or next to it (bullet
direction is not observable, so either side counts).
    s = 2E(x) + E(x-1) + E(x+1) - T(x) + 2B(x) + 2B(x-1) + 2B(x+1) >= 1
(E = fish + enemy sub, T = trail, B = enemy bullet; an enemy at x-1 heading
left leaves its trail on x.)  Row 0 is never reached by enemies.
Goals: fire at enemies ahead in the row, turn toward enemies behind, surface
when oxygen < 60 with a diver aboard, never surface with none, seek divers
vertically, dive from the surface, keep off the spawn columns.
"""
import numpy as np

FRONT, BACK, FB, TR, EB, FISH, ESUB, OXY, DGAUGE, DIVER = range(10)
def idx(y, x, c): return (y * 10 + x) * 10 + c
NIN, BETA = 1000, 20.0
ACTS = ["n", "l", "u", "r", "d", "f"]
DXDY = {"n": (0, 0), "l": (-1, 0), "u": (0, -1), "r": (1, 0), "d": (0, 1)}
cl = lambda v, lo, hi: min(hi, max(lo, v))

rows1, bias1 = [], []
def l1(w, b): rows1.append(w); bias1.append(b); return len(rows1) - 1
def vec(cells):
    w = np.zeros(NIN)
    for (y, x, c), a in cells: w[idx(y, x, c)] += a
    return w
pcol = lambda x: vec([((y, x, FRONT), 1) for y in range(10)])
prow = lambda r: vec([((r, x, FRONT), 1) for x in range(10)])

# L1: RP[(x, x2, r2)] = player in column x AND cell (x2, r2) unsafe   (224)
RP = {}
for r2 in range(1, 9):
    for x in range(10):
        for x2 in (x - 1, x, x + 1):
            if not 0 <= x2 <= 9: continue
            cells = [((r2, x2, FISH), 2), ((r2, x2, ESUB), 2), ((r2, x2, TR), -1), ((r2, x2, EB), 2)]
            for n in (x2 - 1, x2 + 1):
                if 0 <= n <= 9:
                    cells += [((r2, n, FISH), 1), ((r2, n, ESUB), 1), ((r2, n, EB), 2)]
            RP[(x, x2, r2)] = l1(BETA * (vec(cells) + 15 * pcol(x)), BETA * (-0.5 - 15))
PY = [l1(BETA * prow(r), BETA * -0.5) for r in range(9)]                        # 9

# enemy right / left of the player in the player's row (base-3 column weights)
pxb = lambda f: sum(f(x) * pcol(x) for x in range(10))
ER, EL = [], []
for r in range(1, 9):
    w = vec([((r, x, c), 3.0 ** x) for x in range(10) for c in (FISH, ESUB)])
    w -= pxb(lambda px: 3.0 ** (px + 1) - 0.5)
    ER.append(l1(BETA * (w + 1e5 * prow(r)), BETA * -1e5))
    w = vec([((r, x, c), 3.0 ** (9 - x)) for x in range(10) for c in (FISH, ESUB)])
    w -= pxb(lambda px: 3.0 ** (10 - px) - 0.5)
    EL.append(l1(BETA * (w + 1e5 * prow(r)), BETA * -1e5))
# facing right: x(front) - x(back) = +1
FR = l1(BETA * vec([((y, x, FRONT), x) for y in range(10) for x in range(10)] +
                   [((y, x, BACK), -x) for y in range(10) for x in range(10)]), 0.0)
# a diver in some row above / below (base 4 row weights; assumes <= 3 divers per row)
DELTA = 0.5 * 4.0 ** -8; GD = 10 / DELTA
w = vec([((r, x, DIVER), 4.0 ** -r) for r in range(1, 9) for x in range(10)])
w -= 4.0 * prow(0) + sum((4.0 ** -(k - 1) - DELTA) * prow(k) for k in range(1, 9))
DA = l1(GD * w, 0.0)
w = vec([((r, x, DIVER), 4.0 ** -(9 - r)) for r in range(1, 9) for x in range(10)])
w -= 4.0 * prow(8) + sum((4.0 ** -(9 - k - 1) - DELTA) * prow(k) for k in range(0, 8))
DB = l1(GD * w, 0.0)
OXLOW = l1(BETA * -vec([((9, 2, OXY), 1)]), BETA * 0.5)                         # oxygen < 60
HASDIV = l1(BETA * vec([((9, 8, DGAUGE), 1)]), BETA * -0.5)                     # >= 1 diver
L01 = l1(BETA * (pcol(0) + pcol(1)), BETA * -0.5)
R89 = l1(BETA * (pcol(8) + pcol(9)), BETA * -0.5)
H1 = len(rows1); assert H1 <= 256, H1
W1, b1 = np.array(rows1), np.array(bias1)

rows2, bias2 = [], []
def l2(terms, const):          # pre = BETA*(sum c*q(h) + const), q = (h+1)/2
    w = np.zeros(H1); b = const
    for j, c in terms: w[j] += c / 2; b += c / 2
    rows2.append(BETA * w); bias2.append(BETA * b); return len(rows2) - 1
UNSAFE = {a: [] for a in DXDY}
for a, (dx, dy) in DXDY.items():
    for r in range(9):
        if a == "u" and r <= 1: continue           # row 0 target: no enemies
        r2 = cl(r + dy, 0, 8)
        if r2 == 0: continue
        terms = [(RP[(x, cl(x + dx, 0, 9), r2)], 1.0) for x in range(10)] + [(PY[r], 1.0)]
        UNSAFE[a].append(l2(terms, -1.5))
NODIV1 = l2([(PY[1], 1), (HASDIV, -1)], -0.5)
NODIV2 = l2([(PY[2], 1), (HASDIV, -1)], -0.5)
EARLY = l2([(PY[1], 1), (HASDIV, 1), (OXLOW, -1)], -1.5)
SURF = l2([(OXLOW, 1), (HASDIV, 1)], -1.5)
DIVE = l2([(PY[0], 1)], -0.5)
UDA = l2([(DA, 1)], -0.5); UDB = l2([(DB, 1)], -0.5)
AHR = l2([(j, 1) for j in ER] + [(FR, 1)], -1.5)
AHL = l2([(j, 1) for j in EL] + [(FR, -1)], -0.5)
TURNL = l2([(j, 1) for j in EL] + [(FR, 1)], -1.5)
TURNR = l2([(j, 1) for j in ER] + [(FR, -1)], -0.5)
UL01 = l2([(L01, 1)], -0.5); UR89 = l2([(R89, 1)], -0.5)
H2 = len(rows2)
W2, b2 = np.array(rows2), np.array(bias2)

K = 40.0
W3 = np.zeros((6, H2)); b3 = np.zeros(6)
def add(a, j, c): ai = ACTS.index(a); W3[ai, j] += c / 2; b3[ai] += c / 2
for a in DXDY:
    for j in UNSAFE[a]: add(a, j, -K)
for j in UNSAFE["n"]: add("f", j, -K)              # firing does not move the sub
b3[ACTS.index("f")] += 1.0
add("f", AHR, 5); add("f", AHL, 5)
add("l", TURNL, 3); add("r", TURNR, 3)
add("l", UL01, -4); add("r", UR89, -4)
add("u", NODIV1, -K); add("u", NODIV2, -4); add("u", EARLY, -4)
add("u", SURF, 8); add("u", UDA, 2.5); add("d", UDB, 2.5); add("d", DIVE, 4)

if __name__ == "__main__":
    np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print("H1", H1, "H2", H2)
