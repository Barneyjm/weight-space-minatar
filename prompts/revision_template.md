# Prompt for a revision round

Sent to the same Fable subagent that made the previous version, so it keeps its context. Opus got the same information: its own scores and its own failure file from `diagnose_seaquest.py`.

---

Your <previous> Seaquest network scored a mean of <score> over 1,000 episodes (<stats>). That is your number of record.

You now get exactly one revision, which will be scored and reported separately as "revision <k>". I ran your network post hoc on 300 separate diagnostic episodes and wrote the cause of death plus the last 3 frames (board, your logits, chosen vs executed action) for a few examples per cause to `<dir>/failures_v<k>.txt`. Read it.

Same rules as before: read only files inside `<dir>/`, no running the game or the minatar package, no rollouts, no simulator of the dynamics, no training/fitting/search, synthetic hand-built checks against your own reference are fine. Edit the weights by reasoning. Save the result as `<dir>/weights_v<k+1>.npz` (keep earlier weights untouched) with the builder as `write_weights_v<k+1>.py`. Report back in under 200 words: what you changed and why, and your predicted score.
