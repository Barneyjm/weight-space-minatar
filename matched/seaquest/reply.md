**Design.** The policy keeps the submarine parked at column 1 facing right (back cell at column 0), fires by default, and uses per-row conjunction units in layer 1 (tanh thresholds of linear sums) so that all relative-position logic is decided there; layer 2 is an identity pass-through (`W2[j,j]=3`, saturating to ±1), and `W3`/`b3` map each unit to an action with offsets so inactive units contribute ~0.

Hidden units (layer 1, index → meaning):
- 0 `H1`: sub occupies columns {0,1} (ch0 column sums). When false → LEFT (+6).
- 1 `RCUE`: sub occupies {0,1} AND a trail at column 1 AND no friendly bullet anywhere → we are at (0, facing left) → RIGHT (+7), which lands us at (1, facing right); from there the cycle converges to the parked pose.
- 2 `R0`: sub in row 0 → DOWN (+8) (start a dive; row 0 is otherwise safe).
- 3 `SURF`: gauge col 8 lit AND col 2 dark ⇔ oxygen < 60 and ≥1 diver aboard → UP (+12) to resurface safely (never surfaces with 0 divers).
- 4,5 `D1`,`D2`: sub in row 1/2 AND threat in own row (ch1 at cols 0–2, bullets at cols 3–5 via ch1−0.6·ch2) → DOWN (+9). Row 2 never dodges up, so row 1 is only entered at dive start (avoids sticky-"up" surfacing with no diver).
- 6–10 `DU_3..7` / 11–15 `DD_3..7`: same threat conjunction, plus "neighbour row r∓1 has no danger in cols 0–3" → UP / DOWN (+9); direction-aware dodging.
- 16 `D8`: row 8 threat → UP (+9).
- 17–22 `SU_3..8`: sub in rows ≥ r AND a diver (ch2−ch1) in row r−1 → UP (+6.5).
- 23–29 `SD_1..7`: sub in rows 1..r AND a diver in row r+1 → DOWN (+6.5).
- Fire baseline +4, no-op −6. Enemies ahead in the row are killed by the constant fire; threats behind (spawns at column 0) and incoming bullets trigger dodges; divers are sought vertically and picked up as they pass column 1.

```edits
# ---- layer 1 ----
# H1 (unit 0): sub occupies columns 0 and 1
W1[0:600:60, 0] = 8
W1[6:600:60, 0] = 8
b1[0] = -12
# RCUE (unit 1): occupies {0,1} AND trail at col 1 AND no friendly bullet
W1[0:600:60, 1] = 8
W1[6:600:60, 1] = 8
W1[10:600:60, 1] = 8
W1[3:600:6, 1] = -8
b1[1] = -20
# R0 (unit 2): sub in row 0
W1[0:60:6, 2] = 8
b1[2] = -8
# SURF (unit 3): gauge col 8 lit and col 2 dark
W1[593, 3] = 8
W1[557, 3] = -8
b1[3] = -4
# D1 (unit 4): row 1 threat -> down
W1[60:120:6, 4] = 8
W1[61:97:6, 4] = 4
W1[80:98:6, 4] = -2.4
b1[4] = -18.4
# D2 (unit 5): row 2 threat -> down
W1[120:180:6, 5] = 8
W1[121:157:6, 5] = 4
W1[140:158:6, 5] = -2.4
b1[5] = -18.4
# DU_3 (unit 6)
W1[180:240:6, 6] = 8
W1[181:217:6, 6] = 4
W1[200:218:6, 6] = -2.4
W1[121:145:6, 6] = -4
b1[6] = -18.4
# DU_4 (unit 7)
W1[240:300:6, 7] = 8
W1[241:277:6, 7] = 4
W1[260:278:6, 7] = -2.4
W1[181:205:6, 7] = -4
b1[7] = -18.4
# DU_5 (unit 8)
W1[300:360:6, 8] = 8
W1[301:337:6, 8] = 4
W1[320:338:6, 8] = -2.4
W1[241:265:6, 8] = -4
b1[8] = -18.4
# DU_6 (unit 9)
W1[360:420:6, 9] = 8
W1[361:397:6, 9] = 4
W1[380:398:6, 9] = -2.4
W1[301:325:6, 9] = -4
b1[9] = -18.4
# DU_7 (unit 10)
W1[420:480:6, 10] = 8
W1[421:457:6, 10] = 4
W1[440:458:6, 10] = -2.4
W1[361:385:6, 10] = -4
b1[10] = -18.4
# DD_3 (unit 11)
W1[180:240:6, 11] = 8
W1[181:217:6, 11] = 4
W1[200:218:6, 11] = -2.4
W1[241:265:6, 11] = -4
b1[11] = -18.4
# DD_4 (unit 12)
W1[240:300:6, 12] = 8
W1[241:277:6, 12] = 4
W1[260:278:6, 12] = -2.4
W1[301:325:6, 12] = -4
b1[12] = -18.4
# DD_5 (unit 13)
W1[300:360:6, 13] = 8
W1[301:337:6, 13] = 4
W1[320:338:6, 13] = -2.4
W1[361:385:6, 13] = -4
b1[13] = -18.4
# DD_6 (unit 14)
W1[360:420:6, 14] = 8
W1[361:397:6, 14] = 4
W1[380:398:6, 14] = -2.4
W1[421:445:6, 14] = -4
b1[14] = -18.4
# DD_7 (unit 15)
W1[420:480:6, 15] = 8
W1[421:457:6, 15] = 4
W1[440:458:6, 15] = -2.4
W1[481:505:6, 15] = -4
b1[15] = -18.4
# D8 (unit 16): row 8 threat -> up
W1[480:540:6, 16] = 8
W1[481:517:6, 16] = 4
W1[500:518:6, 16] = -2.4
b1[16] = -18.4
# SU_3..SU_8 (units 17..22): sub in rows >= r AND diver in row r-1 -> up
W1[180:540:6, 17] = 8
W1[122:182:6, 17] = 4
W1[121:181:6, 17] = -4
b1[17] = -18.4
W1[240:540:6, 18] = 8
W1[182:242:6, 18] = 4
W1[181:241:6, 18] = -4
b1[18] = -18.4
W1[300:540:6, 19] = 8
W1[242:302:6, 19] = 4
W1[241:301:6, 19] = -4
b1[19] = -18.4
W1[360:540:6, 20] = 8
W1[302:362:6, 20] = 4
W1[301:361:6, 20] = -4
b1[20] = -18.4
W1[420:540:6, 21] = 8
W1[362:422:6, 21] = 4
W1[361:421:6, 21] = -4
b1[21] = -18.4
W1[480:540:6, 22] = 8
W1[422:482:6, 22] = 4
W1[421:481:6, 22] = -4
b1[22] = -18.4
# SD_1..SD_7 (units 23..29): sub in rows 1..r AND diver in row r+1 -> down
W1[60:120:6, 23] = 8
W1[122:182:6, 23] = 4
W1[121:181:6, 23] = -4
b1[23] = -18.4
W1[60:180:6, 24] = 8
W1[182:242:6, 24] = 4
W1[181:241:6, 24] = -4
b1[24] = -18.4
W1[60:240:6, 25] = 8
W1[242:302:6, 25] = 4
W1[241:301:6, 25] = -4
b1[25] = -18.4
W1[60:300:6, 26] = 8
W1[302:362:6, 26] = 4
W1[301:361:6, 26] = -4
b1[26] = -18.4
W1[60:360:6, 27] = 8
W1[362:422:6, 27] = 4
W1[361:421:6, 27] = -4
b1[27] = -18.4
W1[60:420:6, 28] = 8
W1[422:482:6, 28] = 4
W1[421:481:6, 28] = -4
b1[28] = -18.4
W1[60:480:6, 29] = 8
W1[482:542:6, 29] = 4
W1[481:541:6, 29] = -4
b1[29] = -18.4
# ---- layer 2: identity pass-through ----
W2[0, 0] = 3
W2[1, 1] = 3
W2[2, 2] = 3
W2[3, 3] = 3
W2[4, 4] = 3
W2[5, 5] = 3
W2[6, 6] = 3
W2[7, 7] = 3
W2[8, 8] = 3
W2[9, 9] = 3
W2[10, 10] = 3
W2[11, 11] = 3
W2[12, 12] = 3
W2[13, 13] = 3
W2[14, 14] = 3
W2[15, 15] = 3
W2[16, 16] = 3
W2[17, 17] = 3
W2[18, 18] = 3
W2[19, 19] = 3
W2[20, 20] = 3
W2[21, 21] = 3
W2[22, 22] = 3
W2[23, 23] = 3
W2[24, 24] = 3
W2[25, 25] = 3
W2[26, 26] = 3
W2[27, 27] = 3
W2[28, 28] = 3
W2[29, 29] = 3
# ---- layer 3: actions 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire ----
W3[0, 1] = -3
W3[1, 3] = 3.5
W3[2, 4] = 4
W3[3, 2] = 6
W3[4, 4] = 4.5
W3[5, 4] = 4.5
W3[6:11, 2] = 4.5
W3[11:16, 4] = 4.5
W3[16, 2] = 4.5
W3[17:23, 2] = 3.25
W3[23:30, 4] = 3.25
b3[0] = -6
b3[1] = 3
b3[2] = 52.5
b3[3] = 3.5
b3[4] = 58.25
b3[5] = 4
```