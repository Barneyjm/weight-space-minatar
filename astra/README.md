# Re-scoring the authors' Astra networks with our evaluator

The authors publish Astra's networks at
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch/networks).
They are not copied here. `astra_eval.py` scores one of them with the same protocol as `../evaluate.py`: seeds 0–999, the same action-sampling RNG, sticky actions 0.1, and a 10,000-step cap. It feeds the 6-channel input defined in their prompts (`runs/<game>/message.txt`, "THE INPUT").

```
python astra_eval.py breakout path/to/networks/breakout/hand.npz
```

| Game | Authors' harness | Our evaluator | Our mean length |
|---|---|---|---|
| Breakout | 517.4 | 502.9 (se 7.7; 51.7% hit the cap) | 7,188 |
| Asterix | 53.2 | 53.4 (se 1.0) | 864 (theirs: 858) |
| Seaquest | 53.1 | 56.7 (se 1.4) | 1,478 (theirs: 1,412) |

All differences are within about 2 standard errors, as expected from different seeds. So scores from our evaluator are directly comparable to the post's.
