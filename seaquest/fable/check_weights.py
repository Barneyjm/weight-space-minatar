"""Unit check of weights.npz against a plain reference of the intended rule, on hand-built states."""
import numpy as np
SUBF, SUBB, FB, TRAIL, EB, FISH, ESUB, OXY, DIVG, DIVER = range(10)
d = np.load("weights.npz")
W1, b1, W2, b2, W3, b3 = (d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))

def net(state):
    obs = state.reshape(-1).astype(float)
    h1 = np.tanh(obs @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2)
    return h2 @ W3.T + b3

BIN3 = {0: [0, 1, 2], 1: [3, 4, 5, 6], 2: [7, 8, 9]}
def ref(s):
    """Reference rule, written directly from the design (no weight matrices)."""
    ys, xs = np.argwhere(s[:, :, SUBF])[0]
    by, bx = np.argwhere(s[:, :, SUBB])[0]
    facing_right = (xs - bx) == 1
    b2_ = xs // 2
    def hazard(row):
        if row < 1 or row > 8: return False
        bul = any(s[row, xp, EB] for xp in range(max(0, 2*b2_-3), min(9, 2*b2_+4)+1))
        en = any(s[row, xp, FISH] or s[row, xp, ESUB] for xp in range(max(0, 2*b2_-2), min(9, 2*b2_+3)+1))
        return bul or en
    D = hazard(ys); AU = hazard(ys-1); AD = hazard(ys+1)
    b3_ = [b for b, xs_ in BIN3.items() if xs in xs_][0]; lo, hi = BIN3[b3_][0], BIN3[b3_][-1]
    def left(ch): return 1 <= ys <= 8 and any(s[ys, xp, c] for xp in range(0, lo+1) for c in ch)
    def right(ch): return 1 <= ys <= 8 and any(s[ys, xp, c] for xp in range(hi, 10) for c in ch)
    EL, ER, DL, DR = left((FISH, ESUB)), right((FISH, ESUB)), left((DIVER,)), right((DIVER,))
    DROW = 1 <= ys <= 8 and s[ys, :, DIVER].any()
    DAn = ys >= 2 and s[ys-1, :, DIVER].any(); DAf = ys >= 3 and s[1:ys-1, :, DIVER].any()
    DBn = 1 <= ys <= 7 and s[ys+1, :, DIVER].any(); DBf = 1 <= ys <= 6 and s[ys+2:9, :, DIVER].any()
    Vd = s[9, 8, DIVG]; V6 = s[9, 3, DIVG]; ox_ok = s[9, 2, OXY]
    T, R1, R8 = ys == 0, ys == 1, ys == 8
    R1NV = R1 and not Vd; ASC = Vd and not ox_ok; C0, C9 = xs == 0, xs == 9
    z = np.array([3.0, 0, 1, 0, 0, 1])
    if D: z[[0, 1, 3, 5]] -= 100; z[2] += 50; z[4] += 50
    if AU: z[2] -= 60
    if AD: z[4] -= 60
    if R1: z[2] -= 25
    if R1NV: z[2] -= 100
    if R8: z[4] -= 60
    if ASC: z[2] += 30; z[4] -= 30
    if V6: z[2] += 30; z[4] -= 30
    if T: z[4] += 30
    if DL: z[1] += 14
    if DR: z[3] += 14
    if DROW: z[2] -= 14; z[4] -= 14
    if DAn: z[2] += 13
    if DAf: z[2] += 10
    if DBn: z[4] += 13
    if DBf: z[4] += 10
    if ER and facing_right: z[5] += 12
    if EL and not facing_right: z[5] += 12
    if EL and facing_right: z[1] += 12
    if ER and not facing_right: z[3] += 12
    if C0: z[3] += 3
    if C9: z[1] += 3
    return z

def make(sub, facing_right, oxygen=200, divers=0, fish=(), subs=(), bullets=(), divs=()):
    s = np.zeros((10, 10, 10), dtype=bool)
    y, x = sub
    s[y, x, SUBF] = 1; s[y, x - 1 if facing_right else x + 1, SUBB] = 1
    s[9, 0:oxygen * 10 // 200, OXY] = 1
    s[9, 9 - divers:9, DIVG] = 1
    for (fy, fx) in fish: s[fy, fx, FISH] = 1
    for (fy, fx) in subs: s[fy, fx, ESUB] = 1
    for (fy, fx) in bullets: s[fy, fx, EB] = 1
    for (fy, fx) in divs: s[fy, fx, DIVER] = 1
    return s

A = "n l u r d f".split()
cases = {
 "start: surface, dive": make((0, 5), False),
 "surface, row1 hazard near -> wait": make((0, 5), False, fish=[(1, 6)]),
 "fish far right, facing right -> shoot": make((4, 3), True, fish=[(4, 6)]),
 "fish far right, facing left -> turn right": make((4, 3), False, fish=[(4, 7)]),
 "fish far left, facing right -> turn left": make((4, 6), True, fish=[(4, 1)]),
 "fish far left, facing left -> shoot": make((4, 6), False, fish=[(4, 1)]),
 "fish close -> dodge (both ok)": make((4, 3), True, fish=[(4, 5)]),
 "bullet close, row above unsafe -> down": make((4, 3), True, bullets=[(4, 0)], subs=[(3, 4)]),
 "bullet close, row below unsafe -> up": make((4, 3), True, bullets=[(4, 6)], bullets2=None) if False else make((4, 3), True, bullets=[(4, 6), (5, 2)]),
 "row1 no divers, danger -> down only": make((1, 2), True, fish=[(1, 4)]),
 "row1 with diver, danger, row2 unsafe -> surface": make((1, 2), True, divers=1, fish=[(1, 4), (2, 3)]),
 "row8 danger -> up": make((8, 5), True, subs=[(8, 7)]),
 "diver 2 rows up -> up": make((5, 5), True, divs=[(3, 2)]),
 "diver row below -> down": make((5, 5), True, divs=[(6, 9)]),
 "diver in my row right -> right": make((5, 5), True, divs=[(5, 8)]),
 "diver in my row left -> left": make((5, 5), True, divs=[(5, 1)]),
 "diver in my bin -> wait": make((5, 6), True, divs=[(5, 5), (2, 2)]),
 "oxygen low, has diver -> up": make((6, 4), True, oxygen=50, divers=2),
 "oxygen low, no diver -> no ascend": make((6, 4), True, oxygen=50, divers=0),
 "six divers -> up": make((6, 4), True, oxygen=150, divers=6),
 "edge x=0 -> right": make((3, 0), False),
 "edge x=9 -> left": make((3, 9), True),
 "far hazard outside window -> idle": make((4, 2), True, bullets=[(4, 7)], fish=[(4, 8)]),
 "x=6, diver at 5 (in bin) -> wait, no DR": make((5, 6), True, divs=[(5, 5)]),
 "x=3 facing right, fish at 6 -> shoot": make((5, 3), True, fish=[(5, 6)]),
 "x=3 facing right, fish at 5 -> dodge": make((5, 3), True, fish=[(5, 5)]),
 "x=7 facing left, fish at 3 -> shoot": make((5, 7), False, fish=[(5, 3)]),
 "x=0 facing right, fish at 2 -> shoot (D window: fish at 2 is within 2 -> dodge)": make((5, 0), True, fish=[(5, 2)]),
 "enemy sub far left behind, diver right -> seek wins": make((5, 4), True, subs=[(5, 0)], divs=[(5, 9)]),
 "danger + ascend, both rows unsafe": make((4, 3), True, oxygen=40, divers=1, bullets=[(4, 2), (3, 3), (5, 3)]),
}
bad = 0
for name, s in cases.items():
    zn, zr = net(s), ref(s)
    err = np.abs(zn - zr).max()
    ok = err < 1e-3
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name:45s} argmax={A[int(zn.argmax())]}  maxerr={err:.2e}  z={np.round(zn,1)}")
print("all synthetic cases match" if not bad else f"{bad} MISMATCHES")

# saturation check: every layer-1 pre-activation on these states has |pre| >= 10
pres = []
for s in cases.values():
    pres.append(np.abs(s.reshape(-1).astype(float) @ W1.T + b1).min())
print("min |layer-1 preactivation| over cases:", min(pres))
