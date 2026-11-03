import numpy as np
from minatar import Environment
w = np.load("weights.npz")
def fwd(obs):
    h1 = np.tanh(w["W1"] @ obs + w["b1"]); h2 = np.tanh(w["W2"] @ h1 + w["b2"]); return h1, h2, w["W3"] @ h2 + w["b3"]
def fold(u):
    u %= 20; return u if u <= 9 else 19 - u
for seed in range(3):
    e = Environment("breakout", sticky_action_prob=0.0); e.seed(seed); e.reset()
    hist = []
    for t in range(400):
        s = e.state(); obs = np.zeros(600, np.float32); obs[:400] = s.reshape(-1)
        g = e.env
        h1, h2, z = fwd(obs)
        a = int(np.argmax(z))
        D = g.ball_y - g.last_y; dx = g.ball_x - g.last_x
        Fd = h2[27] / 0.01 + 4.5; P = h2[28] / 0.01 + 4.5
        true_d = fold(g.ball_x + dx * (8 - g.ball_y)) if D == 1 else None
        hist.append(f"t{t:3d} ball({g.ball_x},{g.ball_y}) trail({g.last_x},{g.last_y}) D{D:+d} pos{g.pos} Fd{Fd:5.2f} P{P:5.2f} gd{h1[23]:+.0f} ga{h1[24]:+.0f} true_d{true_d} act {'nlurdf'[a]} z_r{z[3]:7.1f} z_l{z[1]:7.1f}")
        r, done = e.act(a)
        if done:
            break
    print(f"--- seed {seed}: ended t={t}"); print("\n".join(hist[-12:]))
