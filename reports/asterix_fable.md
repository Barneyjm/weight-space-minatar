# Asterix — Fable 5.1 one-shot report (verbatim)

Deliverables: `fable_asterix/write_weights.py` (run; builds and saves) and `fable_asterix/weights.npz` (W1 256×600, b1, W2 13×256, b2, W3 6×13, b3; float64, all finite).

**Strategy (reactive dodge-and-collect, read from asterix.py):**
- `Unsafe(r,c)`: enemy on (r,c), or an enemy within 3 cells in row r moving toward c. Direction comes from the trail channel (trail behind entity; x=0/9 without trail = fresh spawn moving inward). One entity per row makes this a clean integer linear threshold (weights 3/2/1 on enemy cells, 1/2/3 on trail cells, 4 on the cell itself; ≥3.5 ⇒ unsafe). Accepted quirk: an enemy at x=0 flags columns 1–3 regardless of direction (same at x=9).
- `T` = player cell Unsafe or enemy horizontally adjacent (any direction). `UU`/`DU` = cell above/below Unsafe (or row 1/8 where the move is a no-op). `GA/GB/GL/GR` = gold anywhere above/below/left/right. Edge-column detectors push the player away from spawn columns 0,1,8,9 (and mildly 2,7); rows 1/8 get a mild push inward.
- Logits: n = 4 − 20T; u = 12T − 30UU + 8GA + 3Row8; d = 12T − 30DU + 8GB + 3Row1; l = 8GL − 8T + 10E89 + 5E7; r symmetric; f = −50. So: calm → stay (93%), chase gold only via safe moves, danger → safe vertical dodge (gold side preferred), trapped → horizontal (inward/gold-biased).

**Layers:** L1 (256 units, gain 8): per-player-cell ANDs T(x,y) [80], UU [70], DU [70], plus row1/row8, GA/GB per row [14], GL/GR per column [16], 4 edge units — all sharp linear thresholds on booleans (margin ≥0.5 ⇒ |tanh| ≥ 0.9993). L2 (13 units): ORs over each group. L3: the logit table with ±1 units remapped to 0/1 via bias.

**Weaknesses:** no horizontal flee direction when trapped (random); coarse gold localization (row/column only, can oscillate); sticky actions can repeat a horizontal step into a chaser; late game (move+spawn every step) will kill it eventually; no budget for distance-4 lookahead.

**Predicted mean score:** ~8–15 (guess; a plain random policy is ~1).

**Checks run (no game code executed):** 20,000 random synthetic states built by my own generator (player anywhere, ≤1 entity per row with its trail, random gold/enemy/direction) compared network logits against a separate semantic reference implementation of the rule: max |logit error| 0.018, zero mismatches. Plus 12 hand-made scenarios (calm, approaching/receding enemies, unsafe row, trapped, gold, edges, spawn at x=0) printed and eyeballed — all behaved as intended.
