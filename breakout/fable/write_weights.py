"""Hand-written weights for MinAtar Breakout (no training).

Strategy: predict the x cell where the ball will be when it is on row 8 just
before the paddle check, and move the paddle there (then hold).  The x path of
the ball is independent of vertical bounces: X(t) = tri(bx + dx*t) with tri the
period-20 triangle wave 0..9,9..0 (wall stall included).  Only the number of
x-steps until the check depends on the vertical geometry:
  moving down:            t = 8 - by
  moving up, bounce at r: t = by + 7 - 2r   (r = lowest brick row on the
                                             diagonal path: 3, 2, 1, or -1 = top wall)
Layer 1: paddle thermometer, direction signs, thermometer staircases of the
unfolded coordinate for each case, and brick lookups along the ball's diagonal
(which brick row the ball will hit first).  Layer 2: for each case a "pos > T"
unit and a "pos < T" unit, gated by direction/case.  Layer 3: left/right/noop.
"""
import numpy as np

OUT = __file__.rsplit('/', 1)[0] + '/weights.npz'
PADDLE, BALL, TRAIL, BRICK = 0, 1, 2, 3
G1, G2, GATE, M = 20.0, 10.0, 200.0, 30.0


def idx(y, x, c):
    return (y * 10 + x) * 4 + c


def tri(u):
    m = u % 20
    return m if m <= 9 else 19 - m


W1, b1, names = [], [], {}


def add1(name, w, b):
    names[name] = len(W1)
    W1.append(np.asarray(w, float))
    b1.append(float(b))
    return names[name]


# ---- paddle thermometer: P_j = +1 iff pos >= j ---------------------------
for j in range(1, 10):
    w = np.zeros(600)
    for x in range(j, 10):
        w[idx(9, x, PADDLE)] = 1
    add1(f'P{j}', G1 * w, G1 * (-0.5))

# ---- direction signs -------------------------------------------------------
# dx: +1 iff trail column < ball column, or equal at x=0 (just reflected)
w = np.zeros(600)
for y in range(10):
    for x in range(10):
        w[idx(y, x, BALL)] = 2 * x + (1 if x == 0 else 0) - (1 if x == 9 else 0)
        w[idx(y, x, TRAIL)] = -2 * x
add1('Dx', G1 * w, 0.0)
# dy: +1 (down) iff trail row < ball row, or equal with ball row < 8
w = np.zeros(600)
for y in range(10):
    for x in range(10):
        w[idx(y, x, BALL)] = 2 * y + (1 if y < 8 else -1)
        w[idx(y, x, TRAIL)] = -2 * y
add1('Dy', G1 * w, 0.0)


# ---- thermometer staircases of tri(u), u linear in the ball one-hot --------
def staircase(name, coef, lo, hi):
    """coef(x, y) -> integer u for ball at (y, x).  Returns (const, [(unit, d)])
    with tri(u) = const + sum d*(h+1)/2 over listed units, valid for lo<=u<=hi."""
    v = np.zeros(600)
    for y in range(10):
        for x in range(10):
            v[idx(y, x, BALL)] = coef(x, y)
    terms = []
    for j in range(lo + 1, hi + 1):
        d = tri(j) - tri(j - 1)
        if d != 0:
            k = add1(f'{name}_ge{j}', G1 * v, G1 * (-j + 0.5))
            terms.append((k, d))
    return tri(lo), terms


stairs = {}
# moving down: t = 8 - by  -> u = bx + dx*(8 - by)
stairs[('down', +1)] = staircase('Ud+', lambda x, y: x + 8 - y, 0, 17)
stairs[('down', -1)] = staircase('Ud-', lambda x, y: x - 8 + y, -8, 9)
# moving up, bounce row r -> offset o = 7 - 2r ; u = bx + dx*(by + o)
UP_CASES = {'r3': (1, 4), 'r2': (3, 3), 'r1': (5, 2), 'wall': (9, 0)}  # offset, min row
for case, (o, ymin) in UP_CASES.items():
    stairs[(case, +1)] = staircase(f'U{case}+', lambda x, y, o=o: x + y + o, ymin + o, 17 + o)
    stairs[(case, -1)] = staircase(f'U{case}-', lambda x, y, o=o: x - y - o, -8 - o, 9 - ymin - o)

# ---- brick lookups along the ball's diagonal -------------------------------
# For dx=+1 the up path has x+y = c constant (unfolded); x at row r is tri(c-r).
# For dx=-1, x-y = c constant; x at row r is tri(c+r).
Bunits = {}  # (case, dx) -> list of unit indices


def diag_cells(dx, c, ymin):
    cells = []
    for y in range(ymin, 9):
        x = c - y if dx == +1 else c + y
        if 0 <= x <= 9:
            cells.append((y, x))
    return cells


def xrow(dx, c, r):
    return tri(c - r) if dx == +1 else tri(c + r)


for dx in (+1, -1):
    for case, need, forbid, ymin in (('r3', 3, (), 4), ('r2', 2, (3,), 3), ('r1', 1, (2, 3), 2)):
        lst = []
        crange = range(ymin, 18) if dx == +1 else range(-8, 10 - ymin)
        for c in crange:
            cells = diag_cells(dx, c, ymin)
            if not cells:
                continue
            w = np.zeros(600)
            for (y, x) in cells:
                w[idx(y, x, BALL)] = 1
            w[idx(need, xrow(dx, c, need), BRICK)] += 1
            for r in forbid:
                w[idx(r, xrow(dx, c, r), BRICK)] -= 1
            lst.append(add1(f'B{case}{"+" if dx > 0 else "-"}_{c}', G1 * w, G1 * (-1.5)))
        Bunits[(case, dx)] = lst

H1 = len(W1)
assert H1 <= 256, H1
W1 = np.stack(W1)
b1 = np.array(b1)

# ---- layer 2: comparison units ---------------------------------------------
W2, b2, L_units, R_units = [], [], [], []
Pidx = [names[f'P{j}'] for j in range(1, 10)]


def add2(sign, const, terms, gate_terms, gate_const, is_left):
    """pre = G2*(sign*(pos - T) - 0.5) + gate.  pos = 4.5 + sum P_j/2.
    T = const + sum d/2 + sum (d/2) h."""
    w = np.zeros(H1)
    b = 0.0
    for p in Pidx:
        w[p] += sign * G2 * 0.5
    b += sign * G2 * 4.5
    Tc = const + sum(d for _, d in terms) / 2.0
    for k, d in terms:
        w[k] += -sign * G2 * d / 2.0
    b += -sign * G2 * Tc
    b += -G2 * 0.5
    for k, cf in gate_terms:
        w[k] += cf
    b += gate_const
    (L_units if is_left else R_units).append(len(W2))
    W2.append(w)
    b2.append(b)


for dx in (+1, -1):
    # gate for direction: GATE*((dx*Dx - 1)/2 + (+-Dy - 1)/2)
    for mode in ('down', 'up'):
        sy = +1 if mode == 'down' else -1
        dir_terms = [(names['Dx'], GATE * dx / 2.0), (names['Dy'], GATE * sy / 2.0)]
        dir_const = -GATE
        if mode == 'down':
            const, terms = stairs[('down', dx)]
            for sign, is_left in ((+1, True), (-1, False)):
                add2(sign, const, terms, dir_terms, dir_const, is_left)
        else:
            for case in UP_CASES:
                const, terms = stairs[(case, dx)]
                gt = list(dir_terms)
                gc = dir_const
                if case == 'wall':
                    # require R3=R2=R1=0 : -GATE*sum (B+1)/2
                    for cs in ('r3', 'r2', 'r1'):
                        for k in Bunits[(cs, dx)]:
                            gt.append((k, -GATE / 2.0))
                            gc += -GATE / 2.0
                else:
                    # require R_case = 1 : GATE*(sum (B+1)/2 - 1)
                    for k in Bunits[(case, dx)]:
                        gt.append((k, GATE / 2.0))
                        gc += GATE / 2.0
                    gc += -GATE
                for sign, is_left in ((+1, True), (-1, False)):
                    add2(sign, const, terms, gt, gc, is_left)

H2 = len(W2)
W2 = np.stack(W2)
b2 = np.array(b2)

# ---- layer 3: actions ['n','l','u','r','d','f'] ----------------------------
W3 = np.zeros((6, H2))
b3 = np.full(6, -M)
b3[0] = 0.0  # noop
nL, nR = len(L_units), len(R_units)
for k in L_units:
    W3[1, k] = M / 2.0
b3[1] = M / 2.0 * (nL - 1)   # one active unit: z = (M/2)*h_active
for k in R_units:
    W3[3, k] = M / 2.0
b3[3] = M / 2.0 * (nR - 1)

np.savez(OUT, W1=W1.astype(np.float64), b1=b1, W2=W2.astype(np.float64), b2=b2,
         W3=W3, b3=b3)
print('saved', OUT, 'H1 =', H1, 'H2 =', H2)
