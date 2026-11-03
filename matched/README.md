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
