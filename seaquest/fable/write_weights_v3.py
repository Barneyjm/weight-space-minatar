"""
Revision 2 of the hand-written MinAtar Seaquest weights (no training/search/rollouts).
Same architecture/format; saves weights_v3.npz.

Changes vs v2 (from failures_v2.txt):
 * 78% of points came from 6-diver bonuses and most deaths were late-game crowding, so survival
   and ramp speed matter more than aimed shooting.  EL/ER + turn/shoot units are dropped; the sub
   simply fires whenever idle (z_f 7 > z_n 6; firing never moves the sub).  The freed units buy a
   third-row soft lookahead AU3/AD3 (-20): a deliberate vertical step whose 2 sticky repeats
   would land on a hazard is no longer taken for seeking, while dodges (50) still can.
 * Flee is wall-aware: at x=0 'l' (and at x=9 'r') is a no-op, so it is suppressed below the
   blocked-vertical level (-175); horizontal block vs a near hazard is -130.
 * Row 2 with no divers blocks 'u' at -45 (dodge up only if down is blocked); the upward diver
   pull from row 3 ignores row-1 divers (unreachable without a diver aboard anyway).
 * Ascend thresholds lowered to oxygen <40 (+30) and <20 (+40 more) to surface ~20% less often
   (each surface ramps the difficulty); no block override any more (it walked into a sub).
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


units, groups = [], {}


def add(group, key, w, b):
    units.append((f"{group}{key}", w, b))
    groups.setdefault(group, {})[group + str(key)] = len(units) - 1


BIN2 = {b: [2 * b, 2 * b + 1] for b in range(5)}
BIN3 = {0: [0, 1, 2], 1: [3, 4, 5, 6], 2: [7, 8, 9]}
HALF = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}
EN = (FISH, ESUB)


def cells(w, row, lo, hi, chans, val=1.0):
    for xp in range(max(0, lo), min(9, hi) + 1):
        for c in chans:
            w[idx(row, xp, c)] += val


def player_cells(w, y, xs, val=PW):
    for x in xs:
        w[idx(y, x, SUBF)] += val


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
    if 1 <= row <= 8:
        cells(w, row, 2 * b - 1, 2 * b + 2, EN); cells(w, row, 2 * b - 2, 2 * b + 3, (EB,))


# AU/AD: fatal-entry windows in the next row and the one after (one sticky repeat)
for y in range(2, 9):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); near_hazard(w, y - 1, b); near_hazard(w, y - 2, b)
        add("AU", (y, b), w, -(PW + 0.5))
for y in range(0, 8):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); near_hazard(w, y + 1, b); near_hazard(w, y + 2, b)
        add("AD", (y, b), w, -(PW + 0.5))
# AU3/AD3: soft lookahead three rows away (two sticky repeats), 3 column bins +-1
for y in range(4, 9):
    for b, xs in BIN3.items():
        w = np.zeros(N_IN); player_cells(w, y, xs); cells(w, y - 3, xs[0] - 1, xs[-1] + 1, EN + (EB,))
        add("AU3", (y, b), w, -(PW + 0.5))
for y in range(0, 6):
    for b, xs in BIN3.items():
        w = np.zeros(N_IN); player_cells(w, y, xs); cells(w, y + 3, xs[0] - 1, xs[-1] + 1, EN + (EB,))
        add("AD3", (y, b), w, -(PW + 0.5))
# DL/DR: diver in my row in the other half of the screen
for y in range(1, 9):
    for h, xs in HALF.items():
        wl = np.zeros(N_IN); player_cells(wl, y, xs)
        if h == 1: cells(wl, y, 0, 4, (DIVER,))
        add("DL", (y, h), wl, -(PW + 0.5))
        wr = np.zeros(N_IN); player_cells(wr, y, xs)
        if h == 0: cells(wr, y, 5, 9, (DIVER,))
        add("DR", (y, h), wr, -(PW + 0.5))


def divers_in_rows(w, rows):
    for r in rows:
        cells(w, r, 0, 9, (DIVER,))


for y in range(2, 9):   # diver above: row 1 only pulls from row 2
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, [1] if y == 2 else range(2, y))
    add("DA", y, w, -(PW + 0.5))
for y in range(1, 8):
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, range(y + 1, 9)); add("DB", y, w, -(PW + 0.5))
for y in range(1, 9):
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, [y]); add("DROW", y, w, -(PW + 0.5))

VD = idx(9, 8, DIVG); V6C = idx(9, 3, DIVG)
OX1, OX0 = idx(9, 1, OXY), idx(9, 0, OXY)    # on iff oxygen >= 40 / 20


def flag(name, cellvals, b):
    w = np.zeros(N_IN)
    for i, v in cellvals: w[i] += v
    add(name, 0, w, b)


rowcells = lambda y, v=1.0: [(idx(y, x, SUBF), v) for x in range(10)]
colcells = lambda x, v=1.0: [(idx(y, x, SUBF), v) for y in range(9)]
flag("T", rowcells(0), -0.5)
flag("R1", rowcells(1), -0.5)
flag("R1NV", rowcells(1) + [(VD, -1)], -0.5)
flag("R2NV", rowcells(2) + [(VD, -1)], -0.5)
flag("R8", rowcells(8), -0.5)
flag("ASC", [(VD, 1), (OX1, -1)], -0.5)
flag("ASC2", [(VD, 1), (OX0, -1)], -0.5)
flag("V6", [(V6C, 1)], -0.5)
flag("C0", colcells(0), -0.5); flag("C9", colcells(9), -0.5)
flag("C1", colcells(1), -0.5); flag("C8", colcells(8), -0.5)

H1 = len(units)
assert H1 <= 256, H1
W1 = np.zeros((H1, N_IN)); b1 = np.zeros(H1)
for i, (_, w, b) in enumerate(units):
    W1[i] = K1 * w; b1[i] = K1 * b

# ---------------------------------------------------------------- layer 2: identity
H2 = H1
W2 = np.eye(H2) * K2; b2 = np.zeros(H2)

# ---------------------------------------------------------------- layer 3
N, L, U, R, Dn, F = range(6)
W3 = np.zeros((6, H2)); b3 = np.zeros(6)


def out(action, name, weight):
    for j in groups[name].values():
        W3[action, j] += weight / 2.0; b3[action] += weight / 2.0


b3[N] += 6.0; b3[F] += 7.0                                  # idle: fire (never moves the sub)
for grp in ("HL", "HR"):
    out(N, grp, -200.0); out(F, grp, -200.0); out(U, grp, 50.0); out(Dn, grp, 50.0)
out(L, "HL", -130.0); out(R, "HR", -130.0)                  # never step toward a nearby hazard
out(L, "HR", 25.0); out(R, "HL", 25.0)                      # flee away if vertical is blocked
out(U, "AU", -150.0); out(Dn, "AD", -150.0)                 # fatal-entry blocks
out(U, "AU3", -20.0); out(Dn, "AD3", -20.0)                 # soft: no seeking toward a 3rd-row hazard
out(U, "R1", -20.0); out(U, "R1NV", -200.0); out(U, "R2NV", -45.0)
out(Dn, "R8", -150.0)
out(U, "ASC", 30.0); out(Dn, "ASC", -10.0); out(U, "ASC2", 40.0)
out(U, "V6", 30.0); out(Dn, "V6", -10.0)
for grp in ("ASC", "V6"):
    out(L, grp, 8.0); out(R, grp, 8.0)                        # walk out from under a block
out(Dn, "T", 30.0)
out(L, "DL", 14.0); out(R, "DR", 14.0)
out(U, "DROW", -14.0); out(Dn, "DROW", -14.0)
out(U, "DA", 13.0); out(Dn, "DB", 13.0)
out(R, "C0", 10.0); out(L, "C0", -175.0)                    # leave the edge; 'l' at x=0 is a no-op
out(L, "C9", 10.0); out(R, "C9", -175.0)
out(L, "C1", -10.0); out(R, "C8", -10.0)

np.savez(HERE / "weights_v3.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print(f"saved weights_v3.npz  H1={H1} H2={H2}")
