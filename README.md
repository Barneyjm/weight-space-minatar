# Hand-written neural network weights for MinAtar games

Can a language model *write* the weights of a network that plays a game well, with no training at all? It only gets to read the game's source code.

The question comes from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). That post reports that a model ("Astra") wrote weights for MinAtar Breakout scoring 517.4, against 15.8 for PPO after 100M frames. This repo is an independent informal test, with two Claude models writing weights blind for three MinAtar games:

- **Opus 5.5** (`claude-opus-5-5`) wrote its weights in the main session, which also organized the test.
- **Fable 5.1** (`claude-fable-5-1`) wrote its weights in a fresh subagent with no prior context, at default effort. Each game's first attempt got a new subagent.

## Rules
- Read only the game source (MinAtar 1.0.15). No web, papers, logs, or other attempts.
- No training, fitting, random search, or sweeps against score.
- No game rollouts or forward passes on real game states. Checking the network is allowed only on hand-built synthetic states, against the author's own reference rule.
- One submission per round, scored once. Revisions are scored and reported separately.

Network format, fixed for all entries:
- `tanh(W1·obs+b1)` → `tanh(W2·h1+b2)` → `W3·h2+b3` → softmax-sampled action.
- Up to 256 units per hidden layer.

## Evaluation
`evaluate.py weights.npz <game>`:
- 1,000 episodes, seeds 0–999.
- Sticky actions 0.1, difficulty ramping on.
- 10,000-step cap per episode.
- Actions sampled with a fixed RNG seed, so results are deterministic.

Every number below reproduces exactly from the files here. `results/scores.csv` has the sha256 of each scored weight file.

```
pip install minatar==1.0.15 numpy
python evaluate.py breakout/fable/weights.npz breakout
```

## Results (mean score over 1,000 episodes)

| Game | Opus 5.5 | Fable 5.1 | Reference |
|---|---|---|---|
| Breakout, one shot | 7.1 | **533.9** (40.8% of episodes hit the step cap) | PPO 15.8 / 25.3 (100M / 500M frames); Astra 517.4 (from the post) |
| Breakout, revision 1 | 23.0 (informal, see notes) | — | |
| Asterix, one shot | 13.0 | **23.1** | |
| Seaquest, one shot | 13.2 | **19.0** | |
| Seaquest, revision 1 | 10.5 | **26.3** | |
| Seaquest, revision 2 | **31.9** (median 25, max 142) | 29.2 (median 34, max 68) | |

Full stats, including each author's own prediction: [`results/scores.csv`](results/scores.csv).

## What we think it shows
- **Breakout is special.** Fable's Breakout network predicts where the ball will cross the paddle row, using the period-20 triangle wave of the ball's x position, and moves there. That is a well-known closed-form controller, and MinAtar's source has been public since 2019, so recall and derivation can't be told apart.
- **Other games:** with no textbook strategy, Fable's lead shrinks from 75× to 1.4–1.8×, and both models score in the tens.
- **The same question applies to Astra's 517.** A high Breakout score shows a model can compile a strategy into weights. It doesn't show it can find one.
- **Revisions from failure traces separate the models more than first attempts do.**
  - *Revision 1:* Fable kept its scoring logic and fixed the cause of death, sticky-action repeats into danger. It went from 19.0 to 26.3, and survival nearly tripled. Opus attacked the same cause by suppressing movement, which cut kills, and dropped from 13.2 to 10.5.
  - *Revision 2:* the diagnostics also showed where points came from. Both models converged on a two-row lookahead against sticky repeats plus surfacing with 6 divers for the oxygen bonus. Opus reached 31.9 and Fable 29.2.
  - The means are close: Opus's is higher, driven by a heavier tail (max 142 vs 68), and Fable's median is higher (34 vs 25). Opus's round 2 is not independent; see the caveats.
- **Fable predicts its own score well once it has feedback.** It predicted ~30 for revision 1 (got 26.3) and ~30 for revision 2 (got 29.2). Neither model predicted its one-shot scores well.
- **Hand-written networks here are readable.** Every unit is an exact threshold on 0/1 inputs, such as "this cell is unsafe" or "enemy ahead", so a failing unit can be found from a trace.

## Caveats
- **This is not the original paper's harness.** The evaluator was written from the post's description, so Fable 533.9 vs Astra 517.4 is roughly a tie, not a ranking.
- **The rules tightened after Breakout.** The rule against writing your own simulator was added from Asterix on, and Opus's Breakout revision predates the formal revision protocol.
- **Information flow:**
  - Opus organized the test, so it knew each Fable score before its next game.
  - Opus saw Fable's one-shot Seaquest report, i.e. its strategy, before writing its own Seaquest revision.
  - Fable never saw anything from Opus.
  - Every Opus entry was scored before the matching Fable report arrived.
  - **Seaquest revision 2 (Opus) is contaminated.** Before writing it, Opus had seen two things from Fable: its revision-1 report, which described a two-row lookahead, and its diagnostic header, which showed most of its points come from 6-diver bonuses. Both ideas appear in Opus's revision 2.
- **No audit by Fable's own transcript is published.** The organizer checked each Fable transcript and found that every tool call stayed inside its folder, with no imports of the game package and no web use.
- **The sample is small:** 3 games, 2 models, at most 3 rounds (Seaquest).

## History
- **Round 1, commit `round-1`:** artifacts collected afterwards from frozen files. All hashes match the scored files, and every weight file rebuilds bit-for-bit from its script.
- **Round 2, commits `Round 2a`–`2c`, tag `round-2`:** recorded live, in this order:
  1. the failure traces given to both authors;
  2. each author's new weights with its prediction, committed *before* scoring;
  3. the scores.

## Layout
```
evaluate.py              scoring harness
diagnose_seaquest.py     post-hoc failure traces (different seeds from scoring)
<game>/<author>/         write_weights*.py (builder), weights*.npz, checks, failures_v*.txt
prompts/                 the prompts given to Fable
reports/                 Fable's reports verbatim; Opus design notes
results/scores.csv       every scored file with hash and stats
```

MinAtar (GPL-3.0) is not included; install it from PyPI.
