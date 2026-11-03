import numpy as np
SUBF, SUBB, FB, TRAIL, EB, FISH, ESUB, OXY, DIVG, DIVER = range(10)
d = np.load("weights_v3.npz")
W1, b1, W2, b2, W3, b3 = (d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))

def net(s):
    h1 = np.tanh(s.reshape(-1).astype(float) @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2)
    return h2 @ W3.T + b3

BIN3 = {0: [0, 1, 2], 1: [3, 4, 5, 6], 2: [7, 8, 9]}
def ref(s):
    ys, xs = np.argwhere(s[:, :, SUBF])[0]; b = xs // 2
    def anyc(row, lo, hi, chans):
        return 1 <= row <= 8 and any(s[row, xp, c] for xp in range(max(0, lo), min(9, hi) + 1) for c in chans)
    EN = (FISH, ESUB); ALL = EN + (EB,)
    HR = anyc(ys, 2*b+1, 2*b+4, EN) or anyc(ys, 2*b+1, 2*b+5, (EB,))
    HL = anyc(ys, 2*b-3, 2*b, EN) or anyc(ys, 2*b-4, 2*b, (EB,))
    near = lambda r: anyc(r, 2*b-1, 2*b+2, EN) or anyc(r, 2*b-2, 2*b+3, (EB,))
    AU = ys >= 2 and (near(ys-1) or near(ys-2)); AD = ys <= 7 and (near(ys+1) or near(ys+2))
    b3_ = [k for k, v in BIN3.items() if xs in v][0]; lo3, hi3 = BIN3[b3_][0], BIN3[b3_][-1]
    AU3 = ys >= 4 and anyc(ys-3, lo3-1, hi3+1, ALL); AD3 = ys <= 5 and anyc(ys+3, lo3-1, hi3+1, ALL)
    DL = xs >= 5 and anyc(ys, 0, 4, (DIVER,)); DR = xs <= 4 and anyc(ys, 5, 9, (DIVER,))
    DROW = 1 <= ys <= 8 and s[ys, :, DIVER].any()
    DA = (ys == 2 and s[1, :, DIVER].any()) or (ys >= 3 and s[2:ys, :, DIVER].any())
    DB = 1 <= ys <= 7 and s[ys+1:9, :, DIVER].any()
    Vd, V6 = s[9, 8, DIVG], s[9, 3, DIVG]; ox1, ox0 = s[9, 1, OXY], s[9, 0, OXY]
    z = np.array([6.0, 0, 0, 0, 0, 7])
    for h in (HL, HR):
        if h: z[[0, 5]] -= 200; z[[2, 4]] += 50
    if HL: z[1] -= 130; z[3] += 25
    if HR: z[3] -= 130; z[1] += 25
    if AU: z[2] -= 150
    if AD: z[4] -= 150
    if AU3: z[2] -= 20
    if AD3: z[4] -= 20
    if ys == 1: z[2] -= 20
    if ys == 1 and not Vd: z[2] -= 200
    if ys == 2 and not Vd: z[2] -= 45
    if ys == 8: z[4] -= 150
    asc, asc2 = Vd and not ox1, Vd and not ox0
    if asc: z[2] += 30; z[4] -= 10; z[[1, 3]] += 8
    if asc2: z[2] += 40
    if V6: z[2] += 30; z[4] -= 10; z[[1, 3]] += 8
    if ys == 0: z[4] += 30
    if DL: z[1] += 14
    if DR: z[3] += 14
    if DROW: z[[2, 4]] -= 14
    if DA: z[2] += 13
    if DB: z[4] += 13
    if xs == 0: z[3] += 10; z[1] -= 175
    if xs == 9: z[1] += 10; z[3] -= 175
    if xs == 1: z[1] -= 10
    if xs == 8: z[3] -= 10
    return z

def make(sub, fr, oxygen=200, divers=0, fish=(), subs=(), bullets=(), divs=()):
    s = np.zeros((10, 10, 10), dtype=bool); y, x = sub
    s[y, x, SUBF] = 1; s[y, x - 1 if fr else x + 1, SUBB] = 1
    s[9, 0:oxygen * 10 // 200, OXY] = 1; s[9, 9 - divers:9, DIVG] = 1
    for (a, c) in fish: s[a, c, FISH] = 1
    for (a, c) in subs: s[a, c, ESUB] = 1
    for (a, c) in bullets: s[a, c, EB] = 1
    for (a, c) in divs: s[a, c, DIVER] = 1
    return s

A = "n l u r d f".split()
cases = {
 "start -> dive": make((0, 5), False),
 "surface, hazard 3 rows down -> dive still (soft)": make((0, 5), False, fish=[(3, 6)]),
 "surface, hazard 2 rows down -> wait": make((0, 5), False, fish=[(2, 6)]),
 "idle -> fire": make((4, 4), True),
 "v2 sticky ex1: diver 3 below but fish 3 rows down near -> no seek": make((2, 4), False, divs=[(5, 5)], fish=[(5, 4), (6, 2)], divers=2, oxygen=190),
 "v2 sticky ex1 variant: fish 3 rows down far -> seek ok": make((2, 4), False, divs=[(5, 5)], fish=[(5, 9)], divers=2, oxygen=190),
 "v2 hit ex3: pinned x=0, bullet right, rows blocked -> vertical": make((3, 0), True, bullets=[(3, 3)], fish=[(2, 1), (4, 1)], divers=4, oxygen=90),
 "x=9 pinned, bullet left -> vertical (not no-op r)": make((3, 9), True, bullets=[(3, 6)], fish=[(2, 8), (4, 8)]),
 "hazard right, rows blocked, not at wall -> flee left": make((4, 4), True, fish=[(4, 7)], bullets=[(3, 4), (5, 5)]),
 "v2 surf ex2: row2 no divers, danger, down ok -> down": make((2, 4), True, fish=[(2, 8)], divs=[(1, 0)]),
 "row2 no divers, danger, down blocked -> flee, then up": make((2, 4), True, fish=[(2, 8)], bullets=[(3, 4)]),
 "row3 no divers, diver in row1 only -> no pull": make((3, 4), True, divs=[(1, 7)]),
 "row3 no divers, diver in row2 -> pull": make((3, 4), True, divs=[(2, 7)]),
 "v2 hit ex1: ASC2 dodge, sub directly above -> down not up": make((3, 6), True, subs=[(2, 6)], fish=[(3, 2), (3, 8)], divers=3, oxygen=10),
 "oxygen 50 -> no ascend yet": make((5, 4), True, divers=2, oxygen=50),
 "oxygen 30 -> ascend": make((5, 4), True, divers=2, oxygen=30),
 "oxygen 10 blocked above -> walk": make((5, 4), True, subs=[(4, 6)], divers=2, oxygen=10),
 "V6 -> up": make((6, 4), True, divers=6, oxygen=150),
 "diver other half -> go": make((5, 2), True, divs=[(5, 7)]),
 "diver in my half -> wait/fire": make((5, 2), True, divs=[(5, 0)]),
 "row8 danger -> up": make((8, 5), True, subs=[(8, 8)]),
 "row1 with diver, danger, down ok -> down": make((1, 3), True, fish=[(1, 6)], divers=1),
 "x=8 diver at 9 -> no step": make((3, 8), True, divs=[(3, 9)]),
 "both sides hazard, rows blocked -> vertical": make((4, 4), True, fish=[(4, 1), (4, 7)], bullets=[(3, 4), (5, 5)]),
 "dodge, 3rd row hazard above -> prefer down": make((5, 4), True, fish=[(5, 7)], subs=[(2, 4)]),
}
bad = 0; pres = []
for name, s in cases.items():
    zn, zr = net(s), ref(s); err = np.abs(zn - zr).max(); ok = err < 1e-4; bad += not ok
    pres.append(np.abs(s.reshape(-1).astype(float) @ W1.T + b1).min())
    print(f"{'OK ' if ok else 'BAD'} {name:62s} argmax={A[int(zn.argmax())]} err={err:.1e} z={np.round(zn,0)}")
print("all match" if not bad else f"{bad} MISMATCHES", "| min |pre1| =", min(pres))
