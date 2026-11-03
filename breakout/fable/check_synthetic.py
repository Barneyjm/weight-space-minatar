"""Sanity check of the hand-made network on synthetic observations (no game code)."""
import numpy as np
d = np.load('weights.npz')
W1, b1, W2, b2, W3, b3 = (d[k] for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))

def idx(y, x, c): return (y*10+x)*4 + c
def tri(u):
    m = u % 20
    return m if m <= 9 else 19 - m

def forward(obs):
    h1 = np.tanh(obs @ W1.T + b1)
    h2 = np.tanh(h1 @ W2.T + b2)
    return h2 @ W3.T + b3

def make_obs(bx, by, lx, ly, pos, bricks):
    s = np.zeros((10, 10, 4))
    s[by, bx, 1] = 1; s[9, pos, 0] = 1; s[ly, lx, 2] = 1; s[:, :, 3] = bricks
    return np.concatenate([s.reshape(-1), np.zeros(200)])

def reference(bx, by, lx, ly, pos, bricks):
    # direction rules (my reading of the source)
    if lx < bx or (lx == bx and bx == 0): dx = 1
    else: dx = -1
    down = (ly < by) or (ly == by and by < 8)
    if down:
        t = 8 - by
    else:
        c = bx + by if dx == 1 else bx - by
        xr = lambda r: tri(c - r) if dx == 1 else tri(c + r)
        if by >= 4 and bricks[3, xr(3)]: r = 3
        elif by >= 3 and bricks[2, xr(2)] and not bricks[3, xr(3)]: r = 2
        elif by >= 2 and bricks[1, xr(1)] and not bricks[2, xr(2)] and not bricks[3, xr(3)]: r = 1
        else: r = -1
        t = by + 7 - 2*r
    T = tri(bx + dx*t)
    return 1 if pos > T else (3 if pos < T else 0), T

rng = np.random.RandomState(0)
maps = [np.zeros((10,10)), np.zeros((10,10))]
maps[0][1:4,:] = 1
for _ in range(6):
    m = np.zeros((10,10)); m[1:4,:] = (rng.rand(3,10) < rng.rand()).astype(float); maps.append(m)
n = bad = 0; minmargin = 1e9
for bricks in maps:
    for by in range(9):
        for bx in range(10):
            for ddx in (-1,0,1):
                for ddy in (-1,0,1):
                    lx, ly = bx+ddx, by+ddy
                    if not (0 <= lx <= 9 and 0 <= ly <= 8): continue
                    if ddx == 0 and bx not in (0, 9): continue  # trail==ball column only at walls
                    for pos in range(10):
                        obs = make_obs(bx, by, lx, ly, pos, bricks)
                        z = forward(obs)
                        a_ref, T = reference(bx, by, lx, ly, pos, bricks)
                        a = int(np.argmax(z))
                        zs = np.sort(z); margin = zs[-1] - zs[-2]
                        minmargin = min(minmargin, margin)
                        n += 1
                        if a != a_ref:
                            bad += 1
                            if bad <= 10: print('MISMATCH', bx, by, lx, ly, pos, 'T', T, 'ref', a_ref, 'net', a, np.round(z,2))
print('checked', n, 'mismatches', bad, 'min logit margin', round(minmargin, 3))
# a few explicit hand cases
b = maps[0]
for (bx,by,lx,ly,pos) in [(0,3,0,3,4),(9,3,9,3,4),(4,8,3,7,4),(4,8,3,7,6),(4,8,3,8,4),(7,4,6,5,2)]:
    z = forward(make_obs(bx,by,lx,ly,pos,b)); print((bx,by,lx,ly,pos), 'T', reference(bx,by,lx,ly,pos,b)[1], 'z', np.round(z,2))
