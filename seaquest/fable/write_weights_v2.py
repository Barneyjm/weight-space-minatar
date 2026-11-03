"""
Revision 1 of the hand-written MinAtar Seaquest weights (no training/search/rollouts).
Same architecture/format as write_weights.py; saves weights_v2.npz.

Changes vs v1, from reading failures_v1.txt:
 * Side-specific danger units HL/HR (hazard in my row on my left/right within ~3-4 cells, bullets
   ~4-5) replace D: they trigger the vertical dodge AND block any horizontal step toward the
   hazard (-100), so diver-chasing / turning can no longer walk up to an enemy where one sticky
   repeat is fatal.  Hazard on one side with both vertical rows blocked -> flee away horizontally.
 * Row-above/below blocks (AU/AD) now cover only truly fatal entries (enemy <=1-2, bullet <=2-3
   cells) but also look two rows ahead (sticky double-step), and weigh -150: stronger than any sum
   of positive drives (dodge 50 + ascend 70 + seek 13).
 * Idle is near-deterministic (n=6, others 0): no random vertical steps.  Row 2 with no divers
   blocks 'u' (-20) so a sticky 'u' at row 1 cannot surface diver-less; released when oxygen is low.
 * Oxygen escalation: <60 +30, <40 +40 more, <20 +100 (overrides the block as a last resort);
   while ascending, a small horizontal drive (8) walks out from under a blocking hazard.
 * Edge columns: +10 to leave x=0/9, -10 to step onto them (spawn-on-top deaths).
 * EL/ER and DL/DR use 2 column halves (divers always move toward the centre, so a diver in my
   half comes to me) to pay for the extra units.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBF, SUBB, FB, TRAIL, EB, FISH, ESUB, OXY, DIVG, DIVER = range(10)
N_IN = 1000
K1, K2 = 20.0, 20.0
PW = 20.0


def idx(y, x, c):
    return (y * 10 + x) * 10 + c


units = []
groups = {}


def add(group, key, w, b):
    units.append((f"{group}{key}", w, b))
    groups.setdefault(group, {})[group + str(key)] = len(units) - 1


BIN2 = {b: [2 * b, 2 * b + 1] for b in range(5)}
HALF = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}


def cells(w, row, lo, hi, chans, val=1.0):
    for xp in range(max(0, lo), min(9, hi) + 1):
        for c in chans:
            w[idx(row, xp, c)] += val


def player_cells(w, y, xs, val=PW):
    for x in xs:
        w[idx(y, x, SUBF)] += val


EN = (FISH, ESUB)
# HR / HL: hazard in my row on my right / left (dodge trigger + horizontal block)
for y in range(1, 9):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b])
        cells(w, y, 2 * b + 1, 2 * b + 4, EN); cells(w, y, 2 * b + 1, 2 * b + 5, (EB,))
        add("HR", (y, b), w, -(PW + 0.5))
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b])
        cells(w, y, 2 * b - 3, 2 * b, EN); cells(w, y, 2 * b - 4, 2 * b, (EB,))
        add("HL", (y, b), w, -(PW + 0.5))


def near_hazard(w, row, b):
    """Fatal-entry window in `row`: enemies within ~1-2 cells, bullets within ~2-3 cells."""
    if 1 <= row <= 8:
        cells(w, row, 2 * b - 1, 2 * b + 2, EN); cells(w, row, 2 * b - 2, 2 * b + 3, (EB,))


# AU: row above (and two above, for a sticky double step) unsafe near me; AD: below
for y in range(2, 9):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); near_hazard(w, y - 1, b); near_hazard(w, y - 2, b)
        add("AU", (y, b), w, -(PW + 0.5))
for y in range(0, 8):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); near_hazard(w, y + 1, b); near_hazard(w, y + 2, b)
        add("AD", (y, b), w, -(PW + 0.5))
# EL/ER, DL/DR: enemy / diver in my row in the other half of the screen
for chans, gl, gr in ((EN, "EL", "ER"), ((DIVER,), "DL", "DR")):
    for y in range(1, 9):
        for h, xs in HALF.items():
            wl = np.zeros(N_IN); player_cells(wl, y, xs)
            if h == 1: cells(wl, y, 0, 4, chans)
            add(gl, (y, h), wl, -(PW + 0.5))
            wr = np.zeros(N_IN); player_cells(wr, y, xs)
            if h == 0: cells(wr, y, 5, 9, chans)
            add(gr, (y, h), wr, -(PW + 0.5))
# O: facing right
w = np.zeros(N_IN)
for y in range(10):
    for x in range(10):
        w[idx(y, x, SUBF)] += x; w[idx(y, x, SUBB)] -= x
add("O", 0, w, 0.0)


def divers_in_rows(w, rows):
    for r in rows:
        cells(w, r, 0, 9, (DIVER,))


for y in range(2, 9):
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, range(1, y)); add("DA", y, w, -(PW + 0.5))
for y in range(1, 8):
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, range(y + 1, 9)); add("DB", y, w, -(PW + 0.5))
for y in range(1, 9):
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, [y]); add("DROW", y, w, -(PW + 0.5))

VD = idx(9, 8, DIVG); V6C = idx(9, 3, DIVG)
OX2, OX1, OX0 = idx(9, 2, OXY), idx(9, 1, OXY), idx(9, 0, OXY)   # on iff oxygen >= 60 / 40 / 20


def flag(name, cellvals, b):
    w = np.zeros(N_IN)
    for i, v in cellvals: w[i] += v
    add(name, 0, w, b)


rowcells = lambda y, v=1.0: [(idx(y, x, SUBF), v) for x in range(10)]
flag("T", rowcells(0), -0.5)
flag("R1", rowcells(1), -0.5)
flag("R1NV", rowcells(1) + [(VD, -1)], -0.5)
flag("R2NV", rowcells(2) + [(VD, -1)], -0.5)
flag("R2NVLOW", rowcells(2) + [(VD, -1), (OX2, -1)], -0.5)
flag("R8", rowcells(8), -0.5)
flag("ASC", [(VD, 1), (OX2, -1)], -0.5)
flag("ASC2", [(VD, 1), (OX1, -1)], -0.5)
flag("ASC3", [(VD, 1), (OX0, -1)], -0.5)
flag("V6", [(V6C, 1)], -0.5)
flag("C0", [(idx(y, 0, SUBF), 1) for y in range(9)], -0.5)
flag("C9", [(idx(y, 9, SUBF), 1) for y in range(9)], -0.5)
flag("C1", [(idx(y, 1, SUBF), 1) for y in range(9)], -0.5)
flag("C8", [(idx(y, 8, SUBF), 1) for y in range(9)], -0.5)

H1 = len(units)
assert H1 <= 256, H1
W1 = np.zeros((H1, N_IN)); b1 = np.zeros(H1)
for i, (_, w, b) in enumerate(units):
    W1[i] = K1 * w; b1[i] = K1 * b

# ---------------------------------------------------------------- layer 2
passthrough = [i for i, (n, _, _) in enumerate(units) if not (n.startswith("EL") or n.startswith("ER"))]
H2 = len(passthrough) + 4
assert H2 <= 256, H2
W2 = np.zeros((H2, H1)); b2 = np.zeros(H2)
h2_of = {}
for j, i in enumerate(passthrough):
    W2[j, i] = K2; h2_of[i] = j
iO = groups["O"]["O0"]
EL_idx = list(groups["EL"].values()); ER_idx = list(groups["ER"].values())


def and_unit(j, idxs, o_sign):
    for i in idxs:
        W2[j, i] = K2 * 0.5
    W2[j, iO] = K2 * 0.5 * o_sign
    b2[j] = K2 * (len(idxs) * 0.5 + 0.5 - 1.5)


base = len(passthrough)
SHOOT_R, SHOOT_L, TURN_L, TURN_R = base, base + 1, base + 2, base + 3
and_unit(SHOOT_R, ER_idx, +1); and_unit(SHOOT_L, EL_idx, -1)
and_unit(TURN_L, EL_idx, +1); and_unit(TURN_R, ER_idx, -1)

# ---------------------------------------------------------------- layer 3
N, L, U, R, Dn, F = range(6)
W3 = np.zeros((6, H2)); b3 = np.zeros(6)


def out(action, h2_units, weight):
    for j in h2_units:
        W3[action, j] += weight / 2.0; b3[action] += weight / 2.0


def g(name):
    return [h2_of[i] for i in groups[name].values()]


b3[N] += 6.0; b3[F] += 1.0
for grp in ("HL", "HR"):
    out(N, g(grp), -200.0); out(F, g(grp), -200.0)
    out(U, g(grp), 50.0); out(Dn, g(grp), 50.0)
out(L, g("HL"), -100.0); out(R, g("HR"), -100.0)      # never step toward a nearby hazard
out(L, g("HR"), 25.0); out(R, g("HL"), 25.0)          # flee away if vertical is blocked
out(U, g("AU"), -150.0); out(Dn, g("AD"), -150.0)     # fatal-entry blocks
out(U, g("R1"), -20.0); out(U, g("R1NV"), -200.0)
out(U, g("R2NV"), -20.0); out(U, g("R2NVLOW"), 20.0)
out(Dn, g("R8"), -150.0)
out(U, g("ASC"), 30.0); out(U, g("ASC2"), 40.0); out(U, g("ASC3"), 100.0)
out(Dn, g("ASC"), -10.0); out(Dn, g("ASC3"), -100.0)   # -10 keeps a safe down-dodge (40) above a horizontal flee (33)
out(U, g("V6"), 30.0); out(Dn, g("V6"), -10.0)
for grp in ("ASC", "V6"):
    out(L, g(grp), 8.0); out(R, g(grp), 8.0)            # walk out from under a block
out(Dn, g("T"), 30.0)
out(L, g("DL"), 14.0); out(R, g("DR"), 14.0)
out(U, g("DROW"), -14.0); out(Dn, g("DROW"), -14.0)
out(U, g("DA"), 13.0); out(Dn, g("DB"), 13.0)
out(F, [SHOOT_R, SHOOT_L], 12.0)
out(L, [TURN_L], 12.0); out(R, [TURN_R], 12.0)
out(R, g("C0"), 10.0); out(L, g("C9"), 10.0)
out(L, g("C1"), -10.0); out(R, g("C8"), -10.0)

np.savez(HERE / "weights_v2.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print(f"saved weights_v2.npz  H1={H1} H2={H2}")
