"""Score the authors' published Astra networks with our evaluator's protocol,
using the 6-channel input defined in their prompts (message.txt, THE INPUT)."""
import sys, numpy as np
from minatar import Environment
game, path = sys.argv[1], sys.argv[2]
z = np.load(path, allow_pickle=False)
W1, b1 = z["actor/dense1/kernel"].T, z["actor/dense1/bias"]
W2, b2 = z["actor/dense2/kernel"].T, z["actor/dense2/bias"]
W3, b3 = z["actor/logits/kernel"].T, z["actor/logits/bias"]
MAP = {  # our channel -> list of MinAtar channels
    "breakout": {0: [0], 2: [3], 3: [1], 4: [2]},
    "asterix": {0: [0], 1: [1], 2: [3], 4: [2]},
    "seaquest": {0: [0, 1], 1: [5, 6, 4], 2: [5, 6, 9], 3: [2], 4: [3, 1], 5: [7, 8]},
}[game]
def encode(st):
    o = np.zeros((10, 10, 6), np.float32)
    for c, src in MAP.items(): o[:, :, c] = st[:, :, src].any(2)
    return o.reshape(-1)
N_EP, MAX = 1000, 10_000; rng = np.random.default_rng(12345)
envs = []
for i in range(N_EP):
    e = Environment(game, sticky_action_prob=0.1); e.seed(i); e.reset(); envs.append(e)
score = np.zeros(N_EP); steps = np.zeros(N_EP, int); alive = np.ones(N_EP, bool)
for t in range(MAX):
    ids = np.flatnonzero(alive)
    if not len(ids): break
    x = np.stack([encode(envs[i].state()) for i in ids])
    lg = np.tanh(np.tanh(x @ W1.T + b1) @ W2.T + b2) @ W3.T + b3
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    acts = (p.cumsum(1) < rng.random((len(ids), 1))).sum(1).clip(0, 5)
    for j, i in enumerate(ids):
        r, d = envs[i].act(int(acts[j])); score[i] += r; steps[i] += 1
        if d: alive[i] = False
print(f"{game}: mean {score.mean():.1f} (se {score.std()/np.sqrt(N_EP):.1f}) median {np.median(score):.0f} max {score.max():.0f} mean length {steps.mean():.0f} cap {(steps>=MAX).mean():.1%}")
