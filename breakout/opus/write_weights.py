"""Hand-written weights for a 600->32->32->6 tanh policy for MinAtar Breakout.

No training, no data, no environment interaction, no forward passes. Every
weight below is set from reasoning about breakout.py.

Input layout (my choice): MinAtar state (10,10,4) bool, flattened in numpy
order -> index (y*10 + x)*4 + c, channels paddle=0, ball=1, trail=2, brick=3.
Inputs 400..599 are zero padding. Actions: n,l,u,r,d,f = 0..5.

Idea
- D = y_ball - y_trail: +1 ball falling, -1 rising, 0 on a bounce frame.
- Falling at row y with horizontal step dx: the column at row 8 is
  fold(x + dx*(8-y)), fold = the game's wall reflection (period 20, the
  ball sits at a wall for two frames). With dx = x - x_trail and
  y_trail = y-1 this is linear in the one-hot ball and trail cells:
  u_d = x*(9-y) - x_t*(7-y_t).
- Rising at row y: assume it bounces off the bottom brick row (row 3), so
  it reaches row 8 after y+1 horizontal steps; at rows 1-3 assume the top
  wall (y+9 steps, written as y-11 mod 20); at row 0, 9 steps.
  u_a = x*(1+m(y)) - x_t*m(y_t-1).
- fold() is built from tanh step units at half-integer thresholds.
- Output: move toward the target column, no-op when on it or on a bounce
  frame.
"""
import numpy as np

N_IN, H1, H2, N_OUT = 600, 32, 32, 6
PADDLE, BALL, TRAIL = 0, 1, 2
A_N, A_L, A_U, A_R, A_D, A_F = range(6)

BETA = 20.0     # step sharpness (inputs are integers, thresholds at k+0.5)
M1 = 1e4        # gate strength in layer 1
M2 = 1e5        # gate strength in layer 2
EPS1 = 0.004    # near-linear carriers in layer 1
EPS2 = 0.01     # near-linear carriers in layer 2
GATE = 20.0
C = 40.0        # output margin scale
K = 2000.0      # no-op override on bounce frames


def idx(y, x, c):
    return (y * 10 + x) * 4 + c


def vec(channel, fn, rows=range(10)):
    v = np.zeros(N_IN)
    for y in rows:
        for x in range(10):
            v[idx(y, x, channel)] = fn(x, y)
    return v


def fold(u):
    u = u % 20
    return u if u <= 9 else 19 - u


def steps(lo, hi):
    """fold(u) for integer u in [lo, hi] = base + sum delta_k * [u > k+0.5]."""
    ks = [(k, fold(k + 1) - fold(k)) for k in range(lo, hi) if fold(k + 1) != fold(k)]
    return fold(lo), ks


def m_rise(y):
    if y == 0:
        return 9
    if y <= 3:
        return y - 11
    return y + 1


D = vec(BALL, lambda x, y: y) - vec(TRAIL, lambda x, y: y)
U_D = vec(BALL, lambda x, y: x * (9 - y)) - vec(TRAIL, lambda x, y: x * (7 - y))
U_A = vec(BALL, lambda x, y: x * (1 + m_rise(y))) - vec(TRAIL, lambda x, y: x * m_rise(y - 1) if y >= 1 else 0)
P = vec(PADDLE, lambda x, y: x, rows=[9])

base_d, ks_d = steps(-7, 16)    # falling: rows 1..8, |dx*(8-y)| <= 7
base_a, ks_a = steps(-10, 19)   # rising: leads in [-10, 9]

W1 = np.zeros((H1, N_IN)); b1 = np.zeros(H1)
W2 = np.zeros((H2, H1)); b2 = np.zeros(H2)
W3 = np.zeros((N_OUT, H2)); b3 = np.zeros(N_OUT)

# ---- layer 1 ----
assert len(ks_d) <= 21
for j, (k, _) in enumerate(ks_d):           # gated fold steps of u_d (falling only)
    W1[j] = BETA * U_D + M1 * D
    b1[j] = -BETA * (k + 0.5) - M1
UA, PN, GD, GA = 21, 22, 23, 24
W1[UA] = EPS1 * U_A; b1[UA] = -EPS1 * 4.5     # u_a carrier
W1[PN] = EPS1 * P;   b1[PN] = -EPS1 * 4.5     # paddle carrier
W1[GD] = GATE * D;   b1[GD] = -GATE * 0.5     # falling gate
W1[GA] = -GATE * D;  b1[GA] = -GATE * 0.5     # rising gate

# ---- layer 2 ----
assert len(ks_a) <= 27
for j, (k, _) in enumerate(ks_a):           # gated fold steps of u_a (rising only)
    W2[j, UA] = BETA / EPS1
    W2[j, GA] = M2
    b2[j] = BETA * (4.5 - k - 0.5) - M2
FD, PC, GD2, GA2 = 27, 28, 29, 30
for j, (k, dk) in enumerate(ks_d):          # carrier for F_d - 4.5
    W2[FD, j] = EPS2 * dk / 2
W2[FD, GD] = EPS2 * base_d / 2
b2[FD] = EPS2 * (base_d / 2 + sum(dk for _, dk in ks_d) / 2 - 4.5)
W2[PC, PN] = EPS2 / EPS1                    # carrier for P - 4.5
W2[GD2, GD] = GATE
W2[GA2, GA] = GATE

# ---- output ----
# V = F_d + F_a - P = (h_FD - h_PC)/EPS2 + base_a*(h_GA2+1)/2 + sum dk*(h_a+1)/2
wV = np.zeros(H2); cV = 0.0
wV[FD] += 1 / EPS2; wV[PC] -= 1 / EPS2
wV[GA2] += base_a / 2; cV += base_a / 2
for j, (k, dk) in enumerate(ks_a):
    wV[j] += dk / 2; cV += dk / 2
# flat = 1 - [fall] - [rise] = -(h_GD2 + h_GA2)/2 ; logits get -K*flat
W3[A_R] = C * wV; b3[A_R] = C * (cV - 0.5)
W3[A_L] = -C * wV; b3[A_L] = C * (-cV - 0.5)
for a in (A_R, A_L):
    W3[a, GD2] += K / 2; W3[a, GA2] += K / 2
b3[A_N] = 0.0
for a in (A_U, A_D, A_F):
    b3[a] = -1e4

np.savez(__file__.replace("write_weights.py", "weights.npz"),
         W1=W1.astype(np.float32), b1=b1.astype(np.float32),
         W2=W2.astype(np.float32), b2=b2.astype(np.float32),
         W3=W3.astype(np.float32), b3=b3.astype(np.float32))
print("desc steps", len(ks_d), "base", base_d, "| asc steps", len(ks_a), "base", base_a)
print("params", W1.size + b1.size + W2.size + b2.size + W3.size + b3.size)
print("finite", all(np.isfinite(a).all() for a in (W1, b1, W2, b2, W3, b3)))
