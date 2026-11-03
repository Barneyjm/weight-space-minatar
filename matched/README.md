# Matched test: Fable 5.1 under Astra's exact conditions

This test repeats the post's Experiment 5 setup as closely as we can:

- **Prompt:** the authors' own prompt for each game, unchanged (`runs/<game>/message.txt` in [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch/runs)). It specifies:
  - the 600→32→32→6 tanh network, with every weight starting at 0;
  - the 6-channel input;
  - no tools, one reply, no feedback;
  - weights written as an `edits` block.
- **Model:** Fable 5.1 (`claude-fable-5-1`) at **max** effort, one fresh subagent per game.
- **Delivery:** the subagent's only allowed tool call is reading `message.txt`. After that it answers in its final message. Astra got the same text directly as its prompt. Each transcript is audited for tool calls.
- **Reading the reply:** `../astra/read_edits.py` parses the last `edits` block. It is a statement parser, not code execution. As a check, it rebuilds all three of Astra's published networks bit for bit from Astra's replies (92/55/122 statements, 815/637/803 nonzero weights).
- **Scoring:** `../astra/astra_eval.py`, our evaluator with the 6-channel input. It reproduces Astra's published scores within about 2 se.

This file and the reader were committed before any reply came back.

## Results

Each reply and its weights were committed before scoring (see git history). Scores use the same seeds and RNG as `../evaluate.py`.

| Game | Fable 5.1, Astra's rules | Astra, same rules (our evaluator) | Fable 5.1, easier setup | PPO 100M / 500M |
|---|---|---|---|---|
| Breakout | 61.3 (`xhigh` retry; median 40) | 502.9 | 533.9 | 15.8 / 25.3 |
| Asterix | **36.2** (max effort; median 35) | 53.4 | 23.1 | 25.8 / 55.2 |
| Seaquest | 0.8 (`xhigh` retry) | 56.7 | 19.0 | 5.6 / 88.9 |

- **Failed attempts:** Breakout and Seaquest at **max** effort failed. Each followed the rules (one Read, nothing else), but its single response hit Claude Code's 64,000-token output limit before any weights were written. Both were rerun once each, as fresh agents at `xhigh`, Astra's effort level, with a note to stay under 60k tokens. Astra's replies used 20–29k output tokens.
- **Run time:** each run took 51–59 minutes of wall time.
- **Breakout:** without tools, Fable wrote a 3-unit reactive tracker (50 statements) that aims one cell ahead of the ball. Its easier-setup network instead predicted where the ball would land, using a few hundred units built and checked in Python.
- **Seaquest:** the network dies in about 100 steps. In 200 post-hoc episodes, 68% were hits and 32% surfaced with no divers. It spends 23% of its time in row 1, where a sticky "up" is fatal.
- **Asterix** is the only game where the matched run beat the easier setup. The design keeps the player in column 5 and only changes rows: dodge approaching enemies, wait for approaching gold.

**Takeaway:** under identical conditions, Astra is far ahead of Fable 5.1 on all three games: 503 vs 61, 53 vs 36, and 57 vs 1. Fable's earlier Breakout result depended on tools and a wider network.
