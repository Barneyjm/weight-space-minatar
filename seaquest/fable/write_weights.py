"""
Hand-written weights for MinAtar Seaquest (no training, no search, no rollouts).

Network: h1 = tanh(obs@W1.T+b1); h2 = tanh(h1@W2.T+b2); z = h2@W3.T+b3; action ~ softmax(z)
obs index = (y*10 + x)*10 + channel.  Channels: 0 sub_front, 1 sub_back, 2 friendly_bullet,
3 trail, 4 enemy_bullet, 5 enemy_fish, 6 enemy_sub, 7 oxygen_gauge (row 9), 8 diver_gauge (row 9),
9 diver.  Actions: 0 n, 1 l, 2 u, 3 r, 4 d, 5 f.

Facts from the source used here:
 * Enemies, enemy bullets and divers only ever occupy rows 1..8 (spawn y = randint(1,9)); they
   never change row.  Row 0 = surface, row 9 = gauges.  Player y in 0..8.
 * Player moves first, then bullets/enemies.  Only same-row collisions can kill, so vertical
   dodging is the universal escape; the destination row must not hold a bullet at distance 1.
 * Surfacing (reaching y=0 after being below) with no divers = terminal; with k<6 divers it
   costs one diver and refills oxygen; with 6 it pays oxygen*10//200.  Each surface ramps
   difficulty, so surface only when oxygen is low (< 60) or 6 divers are aboard.
 * Oxygen gauge cells x=0..k-1 on with k = oxygen*10//200; diver gauge cells 9-count..8 on.
 * Facing: back cell is at x-1 when facing right, x+1 when facing left (never wraps).

Layer 1 (255 units, all exact binary ANDs with margin >= 0.5, gain K1=20):
  D[y,bin2]   player in (y, 2-col bin) AND hazard in row y near the bin  (bullets within ~3-4,
              fish/subs within ~2-3)                                     -> dodge trigger
  AU[y,bin2]  same hazard window in row y-1  -> "row above unsafe";  AD[y,bin2] row y+1
  EL/ER[y,bin3] fish/sub in my row to my left / right (3 column bins)
  DL/DR[y,bin3] diver in my row to my left / right
  DROW[y]     a diver somewhere in my row;  DAn/DAf, DBn/DBf: diver in adjacent / farther rows
  O           facing right (sub_x - back_x);  T at surface; R1, R1NV (row1 & no divers); R8;
  ASC (oxygen<60 & >=1 diver); V6 (6 divers); C0/C9 (at an x edge)
Layer 2 (211 units): identity pass-through (gain 10) plus 4 ANDs: shootR = ER&O, shootL = EL&!O,
  turnL = EL&O, turnR = ER&!O.
Layer 3: priority table of logits (dodge 50 / suppress -100 > hard constraints -60 > ascend or
  dive 30 > in-row seek 14 > vertical seek 13/10 > shoot/turn 12 > idle n=3).
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBF, SUBB, FB, TRAIL, EB, FISH, ESUB, OXY, DIVG, DIVER = range(10)
N_IN = 1000
K1, K2 = 20.0, 20.0
PW = 20.0           # weight on the player one-hot inside AND units (bigger than any count sum)


def idx(y, x, c):
    return (y * 10 + x) * 10 + c


# ---------------------------------------------------------------- layer 1
units = []          # (name, w, b) in "obs units"; pre-activation = K1 * (w.obs + b)
groups = {}         # name -> {key: unit index}


def add(group, key, w, b):
    units.append((f"{group}{key}", w, b))
    groups.setdefault(group, {})[key] = len(units) - 1


BIN2 = {b: [2 * b, 2 * b + 1] for b in range(5)}
BIN3 = {0: [0, 1, 2], 1: [3, 4, 5, 6], 2: [7, 8, 9]}


def hazard_window(w, row, b):
    """Count enemy bullets within [2b-3, 2b+4] and fish/subs within [2b-2, 2b+3] of row `row`."""
    for xp in range(max(0, 2 * b - 3), min(9, 2 * b + 4) + 1):
        w[idx(row, xp, EB)] += 1
    for xp in range(max(0, 2 * b - 2), min(9, 2 * b + 3) + 1):
        w[idx(row, xp, FISH)] += 1
        w[idx(row, xp, ESUB)] += 1


def player_cells(w, y, xs, val=PW):
    for x in xs:
        w[idx(y, x, SUBF)] += val


# D: danger in my row
for y in range(1, 9):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); hazard_window(w, y, b)
        add("D", (y, b), w, -(PW + 0.5))
# AU: hazard near me in the row above (y>=2); AD: row below (y<=7)
for y in range(2, 9):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); hazard_window(w, y - 1, b)
        add("AU", (y, b), w, -(PW + 0.5))
for y in range(0, 8):
    for b in range(5):
        w = np.zeros(N_IN); player_cells(w, y, BIN2[b]); hazard_window(w, y + 1, b)
        add("AD", (y, b), w, -(PW + 0.5))
# EL/ER, DL/DR: thing in my row on my left/right (3 bins, strict windows: left = x' <= bin_start,
# right = x' >= bin_end, so the side is never misattributed; things inside my bin give no signal)
for chans, gl, gr in (((FISH, ESUB), "EL", "ER"), ((DIVER,), "DL", "DR")):
    for y in range(1, 9):
        for b, xs in BIN3.items():
            wl = np.zeros(N_IN); player_cells(wl, y, xs)
            for xp in range(0, xs[0] + 1):
                for c in chans:
                    wl[idx(y, xp, c)] += 1
            add(gl, (y, b), wl, -(PW + 0.5))
            wr = np.zeros(N_IN); player_cells(wr, y, xs)
            for xp in range(xs[-1], 10):
                for c in chans:
                    wr[idx(y, xp, c)] += 1
            add(gr, (y, b), wr, -(PW + 0.5))
# O: facing right  (sub_x - back_x = +1 right, -1 left)
w = np.zeros(N_IN)
for y in range(10):
    for x in range(10):
        w[idx(y, x, SUBF)] += x
        w[idx(y, x, SUBB)] -= x
add("O", 0, w, 0.0)


def divers_in_rows(w, rows):
    for r in rows:
        for x in range(10):
            w[idx(r, x, DIVER)] += 1


for y in range(2, 9):   # diver in the row just above
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, [y - 1]); add("DAn", y, w, -(PW + 0.5))
for y in range(3, 9):   # diver 2+ rows above
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, range(1, y - 1)); add("DAf", y, w, -(PW + 0.5))
for y in range(1, 8):   # diver in the row just below
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, [y + 1]); add("DBn", y, w, -(PW + 0.5))
for y in range(1, 7):   # diver 2+ rows below
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, range(y + 2, 9)); add("DBf", y, w, -(PW + 0.5))
for y in range(1, 9):   # diver somewhere in my row
    w = np.zeros(N_IN); player_cells(w, y, range(10)); divers_in_rows(w, [y]); add("DROW", y, w, -(PW + 0.5))

VD = idx(9, 8, DIVG)    # >= 1 diver aboard
V6C = idx(9, 3, DIVG)   # 6 divers aboard
OX2 = idx(9, 2, OXY)    # on iff oxygen >= 60

w = np.zeros(N_IN); player_cells(w, 0, range(10), 1.0); add("T", 0, w, -0.5)
w = np.zeros(N_IN); player_cells(w, 1, range(10), 1.0); add("R1", 0, w, -0.5)
w = np.zeros(N_IN); player_cells(w, 1, range(10), 1.0); w[VD] -= 1; add("R1NV", 0, w, -0.5)
w = np.zeros(N_IN); player_cells(w, 8, range(10), 1.0); add("R8", 0, w, -0.5)
w = np.zeros(N_IN); w[VD] += 1; w[OX2] -= 1; add("ASC", 0, w, -0.5)
w = np.zeros(N_IN); w[V6C] += 1; add("V6", 0, w, -0.5)
w = np.zeros(N_IN)
for y in range(9): w[idx(y, 0, SUBF)] += 1
add("C0", 0, w, -0.5)
w = np.zeros(N_IN)
for y in range(9): w[idx(y, 9, SUBF)] += 1
add("C9", 0, w, -0.5)

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
h2_of = {}  # layer-1 index -> layer-2 index
for j, i in enumerate(passthrough):
    W2[j, i] = K2; h2_of[i] = j

iO = groups["O"][0]
EL_idx = list(groups["EL"].values()); ER_idx = list(groups["ER"].values())


def and_unit(j, idxs, o_sign):
    """h2[j] = AND(any of idxs on, facing right if o_sign>0 else left) in +-1 semantics.
    {0,1} rule: sum((h+1)/2 over idxs) + (1 +- h_O)/2 - 1.5 >= 0.5 when on, <= -0.5 when off."""
    n = len(idxs)
    for i in idxs:
        W2[j, i] = K2 * 0.5
    W2[j, iO] = K2 * 0.5 * o_sign
    b2[j] = K2 * (n * 0.5 + 0.5 - 1.5)


base = len(passthrough)
SHOOT_R, SHOOT_L, TURN_L, TURN_R = base, base + 1, base + 2, base + 3
and_unit(SHOOT_R, ER_idx, +1)
and_unit(SHOOT_L, EL_idx, -1)
and_unit(TURN_L, EL_idx, +1)
and_unit(TURN_R, ER_idx, -1)

# ---------------------------------------------------------------- layer 3
N, L, U, R, Dn, F = range(6)
W3 = np.zeros((6, H2)); b3 = np.zeros(6)


def out(action, h2_units, weight):
    """Add `weight` to logit when the (binary) unit is on: w*(h+1)/2."""
    for j in h2_units:
        W3[action, j] += weight / 2.0
        b3[action] += weight / 2.0


def g(name):
    return [h2_of[i] for i in groups[name].values()]


b3[N] += 3.0; b3[U] += 1.0; b3[F] += 1.0          # idle preferences
for a in (N, L, R, F):
    out(a, g("D"), -100.0)                          # danger: never stay/shoot/sidestep
out(U, g("D"), 50.0); out(Dn, g("D"), 50.0)        # danger: dodge vertically
out(U, g("AU"), -60.0); out(Dn, g("AD"), -60.0)    # never move into a hazardous row
out(U, g("R1"), -25.0)                             # do not surface casually
out(U, g("R1NV"), -100.0)                          # surfacing with no diver is death
out(Dn, g("R8"), -60.0)                            # down at row 8 is a wasted move
out(U, g("ASC"), 30.0); out(Dn, g("ASC"), -30.0)   # low oxygen with a diver: surface
out(U, g("V6"), 30.0); out(Dn, g("V6"), -30.0)     # 6 divers: cash in
out(Dn, g("T"), 30.0)                              # at the surface: dive (AD[0,*] blocks it)
out(L, g("DL"), 14.0); out(R, g("DR"), 14.0)       # diver in my row: go get it
out(U, g("DROW"), -14.0); out(Dn, g("DROW"), -14.0)
out(U, g("DAn"), 13.0); out(U, g("DAf"), 10.0)     # diver above / below
out(Dn, g("DBn"), 13.0); out(Dn, g("DBf"), 10.0)
out(F, [SHOOT_R, SHOOT_L], 12.0)                   # enemy ahead in my row: fire
out(L, [TURN_L], 12.0); out(R, [TURN_R], 12.0)     # enemy behind me: turn toward it
out(R, g("C0"), 3.0); out(L, g("C9"), 3.0)         # drift off the spawn columns

np.savez(HERE / "weights.npz", W1=W1.astype(np.float64), b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print(f"saved weights.npz  H1={H1} H2={H2}")
