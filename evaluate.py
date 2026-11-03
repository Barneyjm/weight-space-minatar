"""Score weights.npz on MinAtar Breakout: 1,000 episodes, sampled actions,
sticky actions 0.1, max 10,000 steps (the post's protocol)."""
import sys, time
import numpy as np
from minatar import Environment

w = np.load(sys.argv[1]); GAME = sys.argv[2]
N_EP, MAX_STEPS = 1000, 10_000
rng = np.random.default_rng(12345)

envs = []
for i in range(N_EP):
    e = Environment(GAME, sticky_action_prob=0.1)
    e.seed(i)
    e.reset()
    envs.append(e)

def policy(obs):  # obs: (n, 600)
    h1 = np.tanh(obs @ w["W1"].T + w["b1"])
    h2 = np.tanh(h1 @ w["W2"].T + w["b2"])
    z = h2 @ w["W3"].T + w["b3"]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    u = rng.random((len(obs), 1))
    return (p.cumsum(1) < u).sum(1).clip(0, 5)

score = np.zeros(N_EP); steps = np.zeros(N_EP, int); alive = np.ones(N_EP, bool)
t0 = time.time()
for t in range(MAX_STEPS):
    ids = np.flatnonzero(alive)
    if len(ids) == 0:
        break
    obs = np.zeros((len(ids), w["W1"].shape[1]), np.float32)
    for j, i in enumerate(ids):
        s = envs[i].state().reshape(-1); obs[j, :s.size] = s
    acts = policy(obs)
    for j, i in enumerate(ids):
        r, done = envs[i].act(int(acts[j]))
        score[i] += r; steps[i] += 1
        if done:
            alive[i] = False
print(f"episodes {N_EP}  mean score {score.mean():.1f}  (sd {score.std():.1f}, se {score.std()/np.sqrt(N_EP):.1f})")
print(f"median {np.median(score):.0f}  min {score.min():.0f}  max {score.max():.0f}")
print(f"hit 10k-step cap: {(steps >= MAX_STEPS).mean():.1%}  mean length {steps.mean():.0f}")
print(f"elapsed {time.time()-t0:.0f}s")
