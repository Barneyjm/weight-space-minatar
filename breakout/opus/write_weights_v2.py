"""Attempt 2 (written after seeing attempt 1 fail; still no training/fitting).

Same 600->32->32->6 tanh network and input layout as attempt 1.

Four cases, each with a linear landing-column estimate u (unfolded across
walls) from the one-hot ball and trail cells:
  fall      D=+1             u = x + dx*(8-y)
  rise      D=-1             u = x + dx*m(y)   (brick row 3 / top wall)
  flat-low  D=0, row <= 4    u = x + dx*(8-y)  (brick or top bounce: now falling)
  flat-high D=0, row 8       u = x + 9*dx      (paddle bounce: now rising)
With w1 = u - P and w2 = u + P (P = paddle column), for u in [-10, 19]:
  fold(u) > P  <=>  (w1 > .5 and w2 < 18.5) or w2 < -1.5
  fold(u) < P  <=>  (w1 < -.5 and w2 > -.5) or w2 > 19.5
Layer 1: those six threshold units per case + case gates.
Layer 2: AND of thresholds with the case gate.  Output: right/left/no-op.
"""
import numpy as np

N_IN, H1, H2, N_OUT = 600, 32, 32, 6
PADDLE, BALL, TRAIL = 0, 1, 2
A_N, A_L, A_U, A_R, A_D, A_F = range(6)
BETA, C = 20.0, 40.0


def idx(y, x, c):
    return (y * 10 + x) * 4 + c


def vec(channel, fn, rows=range(10)):
    v = np.zeros(N_IN)
    for y in rows:
        for x in range(10):
            v[idx(y, x, channel)] = fn(x, y)
    return v


def m_rise(y):
    if y == 0:
        return 9
    if y <= 3:
        return y - 11
    return y + 1


D = vec(BALL, lambda x, y: y) - vec(TRAIL, lambda x, y: y)
P = vec(PADDLE, lambda x, y: x, rows=[9])
LOWROW = vec(BALL, lambda x, y: 1.0 if y <= 4 else 0.0)
ROW8 = vec(BALL, lambda x, y: 1.0 if y == 8 else 0.0)

U = {
    # trail at row y_t = y-1:  u = x(9-y) - x_t(8-y) = x(9-y) - x_t(7-y_t)
    "fall": vec(BALL, lambda x, y: x * (9 - y)) - vec(TRAIL, lambda x, y: x * (7 - y)),
    # trail at row y_t = y+1:  u = x(1+m(y)) - x_t m(y_t-1)
    "rise": vec(BALL, lambda x, y: x * (1 + m_rise(y))) - vec(TRAIL, lambda x, y: x * m_rise(y - 1) if y >= 1 else 0),
    # trail on the same row:   u = x(9-y) - x_t(8-y)
    "flat_low": vec(BALL, lambda x, y: x * (9 - y)) - vec(TRAIL, lambda x, y: x * (8 - y)),
    # same row, row 8:         u = 10x - 9x_t
    "flat_high": vec(BALL, lambda x, y: 10 * x) - vec(TRAIL, lambda x, y: 9 * x),
}
# (vector of w, sign, threshold): unit is on when sign*(w - thr) > 0
THRESH = [("w1", +1, 0.5), ("w2", -1, 18.5), ("w2", -1, -1.5),   # s1 s2 s3
          ("w1", -1, -0.5), ("w2", +1, -0.5), ("w2", +1, 19.5)]   # t1 t2 t3

W1 = np.zeros((H1, N_IN)); b1 = np.zeros(H1)
W2 = np.zeros((H2, H1)); b2 = np.zeros(H2)
W3 = np.zeros((N_OUT, H2)); b3 = np.zeros(N_OUT)

unit = {}
n = 0
for case, u in U.items():
    w = {"w1": u - P, "w2": u + P}
    for i, (name, sign, thr) in enumerate(THRESH):
        W1[n] = BETA * sign * w[name]; b1[n] = -BETA * sign * thr
        unit[(case, i)] = n; n += 1
GD, GA, LOW, HIGH = n, n + 1, n + 2, n + 3
W1[GD] = BETA * D;      b1[GD] = -BETA * 0.5
W1[GA] = -BETA * D;     b1[GA] = -BETA * 0.5
W1[LOW] = BETA * LOWROW; b1[LOW] = -BETA * 0.5
W1[HIGH] = BETA * ROW8;  b1[HIGH] = -BETA * 0.5
assert HIGH < H1

# Layer 2: a 0/1 quantity q = (h+1)/2 enters a unit as weight/2 on h plus bias/2.
def and_unit(j, terms, const=0.0):
    """unit j on iff sum of 0/1 terms (each (index, coef)) + const > 0, BETA sharp."""
    total = const
    for i, coef in terms:
        W2[j, i] += BETA * coef / 2
        total += coef / 2
    b2[j] = BETA * total

def gate_terms(case):
    # 0/1 gate expressed through layer-1 gate units
    # returns (terms, const) so that sum(coef*q) + const == need when open
    if case == "fall":
        return [(GD, 1.0)], 0.0
    if case == "rise":
        return [(GA, 1.0)], 0.0
    # flat = 1 - [fall] - [rise]; flat + row == 2 when open
    row = LOW if case == "flat_low" else HIGH
    return [(GD, -1.0), (GA, -1.0), (row, 1.0)], 1.0

j = 0
right, left = [], []
for case in U:
    g, gconst = gate_terms(case)
    need = 1.0 if case in ("fall", "rise") else 2.0           # gate contributes this when open
    s1, s2, s3, t1, t2, t3 = (unit[(case, i)] for i in range(6))
    for out, a, b in ((right, s1, s2), (left, t1, t2)):       # AND(a, b, gate)
        and_unit(j, [(a, 1.0), (b, 1.0)] + g, const=gconst - (2 + need) + 0.5); out.append(j); j += 1
    for out, a in ((right, s3), (left, t3)):                  # AND(a, gate)
        and_unit(j, [(a, 1.0)] + g, const=gconst - (1 + need) + 0.5); out.append(j); j += 1
assert j <= H2

for a_out, units in ((A_R, right), (A_L, left)):
    for k in units:                                           # C * sum of 0/1 units
        W3[a_out, k] = C / 2; b3[a_out] += C / 2
b3[A_N] = C / 2
for a in (A_U, A_D, A_F):
    b3[a] = -1e4

np.savez(__file__.replace("write_weights_v2.py", "weights_v2.npz"),
         **{k: v.astype(np.float32) for k, v in dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items()})
print("layer1 units used", HIGH + 1, "| layer2 units used", j)
print("finite", all(np.isfinite(a).all() for a in (W1, b1, W2, b2, W3, b3)))
