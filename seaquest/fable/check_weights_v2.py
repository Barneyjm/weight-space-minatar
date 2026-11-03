import numpy as np
SUBF, SUBB, FB, TRAIL, EB, FISH, ESUB, OXY, DIVG, DIVER = range(10)
d = np.load("weights_v2.npz")
W1, b1, W2, b2, W3, b3 = (d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))

def net(s):
    h1 = np.tanh(s.reshape(-1).astype(float) @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2)
    return h2 @ W3.T + b3

def ref(s):
    ys, xs = np.argwhere(s[:, :, SUBF])[0]; by, bx = np.argwhere(s[:, :, SUBB])[0]
    fr = (xs - bx) == 1; b = xs // 2
    def anyc(row, lo, hi, chans):
        return 1 <= row <= 8 and any(s[row, xp, c] for xp in range(max(0, lo), min(9, hi) + 1) for c in chans)
    EN = (FISH, ESUB)
    HR = anyc(ys, 2*b+1, 2*b+4, EN) or anyc(ys, 2*b+1, 2*b+5, (EB,))
    HL = anyc(ys, 2*b-3, 2*b, EN) or anyc(ys, 2*b-4, 2*b, (EB,))
    near = lambda r: anyc(r, 2*b-1, 2*b+2, EN) or anyc(r, 2*b-2, 2*b+3, (EB,))
    AU = ys >= 2 and (near(ys-1) or near(ys-2)); AD = ys <= 7 and (near(ys+1) or near(ys+2))
    inrow = lambda chans, lo, hi: 1 <= ys <= 8 and anyc(ys, lo, hi, chans)
    EL = xs >= 5 and inrow(EN, 0, 4); ER = xs <= 4 and inrow(EN, 5, 9)
    DL = xs >= 5 and inrow((DIVER,), 0, 4); DR = xs <= 4 and inrow((DIVER,), 5, 9)
    DROW = 1 <= ys <= 8 and s[ys, :, DIVER].any()
    DA = ys >= 2 and s[1:ys, :, DIVER].any(); DB = 1 <= ys <= 7 and s[ys+1:9, :, DIVER].any()
    Vd, V6 = s[9, 8, DIVG], s[9, 3, DIVG]; ox2, ox1, ox0 = s[9, 2, OXY], s[9, 1, OXY], s[9, 0, OXY]
    z = np.array([6.0, 0, 0, 0, 0, 1])
    for h in (HL, HR):
        if h: z[[0, 5]] -= 200; z[[2, 4]] += 50
    if HL: z[1] -= 100; z[3] += 25
    if HR: z[3] -= 100; z[1] += 25
    if AU: z[2] -= 150
    if AD: z[4] -= 150
    if ys == 1: z[2] -= 20
    if ys == 1 and not Vd: z[2] -= 200
    if ys == 2 and not Vd: z[2] -= 20
    if ys == 2 and not Vd and not ox2: z[2] += 20
    if ys == 8: z[4] -= 150
    asc, asc2, asc3 = Vd and not ox2, Vd and not ox1, Vd and not ox0
    if asc: z[2] += 30; z[4] -= 10; z[[1, 3]] += 8
    if asc2: z[2] += 40
    if asc3: z[2] += 100; z[4] -= 100
    if V6: z[2] += 30; z[4] -= 10; z[[1, 3]] += 8
    if ys == 0: z[4] += 30
    if DL: z[1] += 14
    if DR: z[3] += 14
    if DROW: z[[2, 4]] -= 14
    if DA: z[2] += 13
    if DB: z[4] += 13
    if (ER and fr) or (EL and not fr): z[5] += 12
    if EL and fr: z[1] += 12
    if ER and not fr: z[3] += 12
    if xs == 0: z[3] += 10
    if xs == 9: z[1] += 10
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
 "surface, hazard 2 rows down near -> wait": make((0, 5), False, fish=[(2, 6)]),
 "idle mid-water -> n": make((4, 4), True),
 "fish 4 right, facing right -> dodge (HR), r blocked": make((4, 3), True, fish=[(4, 6)]),
 "fish 7 right, facing right -> shoot": make((4, 2), True, fish=[(4, 7)]),
 "fish far right, facing left -> turn right": make((4, 2), False, fish=[(4, 8)]),
 "fish far left, facing right -> turn left": make((4, 7), True, fish=[(4, 1)]),
 "v1 sticky ex2: diver right, fish at +2 -> HR blocks r, dodge": make((8, 4), True, fish=[(8, 6)], divs=[(8, 8)], divers=4, oxygen=190),
 "v1 sticky ex2 earlier: fish at +3 from x=3 -> no chase": make((8, 3), True, fish=[(8, 6)], divs=[(8, 8)], divers=4, oxygen=190),
 "v1 hit ex3: sub directly above, fish right, ASC -> down not up": make((3, 2), True, subs=[(2, 2)], fish=[(3, 4)], divers=3, oxygen=50),
 "v1 hit ex1: V6+ASC, fish diag above -> wait (u blocked)": make((6, 1), False, fish=[(5, 1), (7, 4)], divers=6, oxygen=50),
 "v1 sticky ex1: fish 2 rows below -> d blocked": make((1, 3), True, fish=[(3, 3)], divers=4, divs=[(8, 0)]),
 "row2 no divers, diver in row1 -> no chase": make((2, 4), True, divs=[(1, 8)]),
 "row2 no divers, low oxygen, diver row1 -> chase": make((2, 4), True, divs=[(1, 8)], oxygen=50),
 "row1 no divers danger -> down": make((1, 2), True, fish=[(1, 4)]),
 "row1 no divers danger, down blocked -> flee left": make((1, 4), True, fish=[(1, 6), (2, 4)]),
 "ASC blocked above -> walk": make((5, 4), True, subs=[(4, 6)], divers=2, oxygen=50),
 "ASC3 blocked above -> up anyway": make((5, 4), True, subs=[(4, 6)], divers=2, oxygen=10),
 "x=9 -> left strongly": make((3, 9), True),
 "x=8 diver at 9 -> do not step": make((3, 8), True, divs=[(3, 9)]),
 "danger both sides both rows blocked -> vertical": make((4, 4), True, fish=[(4, 1), (4, 7)], bullets=[(3, 4), (5, 5)]),
 "hazard right, both rows blocked -> flee left": make((4, 4), True, fish=[(4, 7)], bullets=[(3, 4), (5, 5)]),
 "row8 danger -> up": make((8, 5), True, subs=[(8, 8)]),
 "six divers -> up": make((6, 4), True, oxygen=150, divers=6),
 "diver in my half -> wait": make((5, 2), True, divs=[(5, 0)]),
 "diver other half -> go": make((5, 2), True, divs=[(5, 7)]),
 "diver above and in-row diver -> in-row wins": make((5, 2), True, divs=[(5, 7), (2, 2)]),
}
bad = 0; pres = []
for name, s in cases.items():
    zn, zr = net(s), ref(s); err = np.abs(zn - zr).max(); ok = err < 1e-4; bad += not ok
    pres.append(np.abs(s.reshape(-1).astype(float) @ W1.T + b1).min())
    print(f"{'OK ' if ok else 'BAD'} {name:60s} argmax={A[int(zn.argmax())]} err={err:.1e} z={np.round(zn,0)}")
print("all match" if not bad else f"{bad} MISMATCHES", "| min |pre1| =", min(pres))
