# Prompt given to Fable 5.1 (one-shot rounds)

Sent verbatim for each game, with `<GAME>`, `<dir>`, the source file name and the
input width filled in. Each game ran in a fresh subagent with no prior context.
`<dir>` was an empty folder holding only a copy of the MinAtar game source.

Variation by game:

- **Breakout** (first test): the rule about writing your own simulator was not yet spelled out. The prompt said PPO gets about 15.8 after 100M frames and 25.3 after 500M.
- **Asterix and Seaquest:** no reference scores were given. The no-simulator rule was added.
- **Seaquest** used the full 1000-input network. Breakout and Asterix used 600 inputs, the last 200 always zero.

---

You are taking a "weight-space programming" test: write the weights of a small neural network by hand, reasoning only from the game's source code, so that it plays MinAtar <GAME> well. No training, no fitting, no search, no running the game.

## Working directory
`<dir>/`

The game source is in `<dir>/minatar_src/` (`<game>.py`, `environment.py`). Read it carefully; every mechanic you need is there.

A Python with numpy is at `../.venv/bin/python` (relative to `<dir>/`). Use it only to build and save the weight arrays and to unit-check your own hand-computed arithmetic on hand-made inputs you construct yourself, never on states produced by the game.

## Rules (strict — this is a no-cheating test)
- Read ONLY files inside `<dir>/`. Do not read, list or grep anything else in the `wsp/` folder or elsewhere in the scratchpad (there are other attempts there; looking at them is cheating). Do not import or execute the installed `minatar` package, and do not execute `<dir>/minatar_src/` code.
- No web access, no looking up existing solutions, papers or logs.
- No gradient descent, no fitting, no random search, no hyperparameter sweeps against any score.
- No game rollouts, no forward passes of your network on game states, and no simulator of the game dynamics written by you either. You may check your network on hand-built synthetic observation arrays you construct from your own reading of the source, compared against a plain reference implementation of your own intended decision rule, to catch arithmetic/sign bugs. Say in your report how much of this you did.
- Submit exactly once. I will score it; you get no feedback.

## Network format (fixed)
Save `<dir>/weights.npz` with float arrays `W1` (H1×<N>), `b1` (H1), `W2` (H2×H1), `b2` (H2), `W3` (6×H2), `b3` (6). Any H1, H2 ≤ 256. The evaluator computes:
```
h1 = tanh(obs @ W1.T + b1)
h2 = tanh(h1 @ W2.T + b2)
z  = h2 @ W3.T + b3
action ~ softmax(z)   # sampled, not argmax
```
`obs` is the 10×10×C boolean state, C-order (index = (y*10 + x)*C + channel), zero-padded to <N>. Action index = position in the environment's action map. Sticky actions with probability 0.1 are on, difficulty ramping is on. Scoring: mean score over 1,000 episodes, capped at 10,000 steps each.

## Deliverables
1. `<dir>/write_weights.py` that builds and saves `<dir>/weights.npz` (run it).
2. A report back to me (under 300 words): the strategy the network implements, how the layers compute it, any known weaknesses, your predicted mean score, and exactly what checks you ran on synthetic inputs.
