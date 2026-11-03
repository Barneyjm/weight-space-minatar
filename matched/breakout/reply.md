**Design.** The policy is a wall-aware ball tracker that always catches the ball on the paddle cell itself (so the ball keeps its x-direction and sweeps every brick cell within 180 steps) and leads the ball by one cell elsewhere so that sticky actions never leave the paddle more than one cell behind.

Layer 1 (three units, all others zero):
- unit 0: q_full = 2·bx − tx + [bx=0] − [bx=9] − px (= target − paddle, where target = ball x + direction from the trail, with the wall clamp handled by the ±1 edge terms), scaled by 0.02 so tanh is linear.
- unit 1: q_plain = bx − px, scaled by 0.02 (used only when the ball is on row 8, i.e. about to drop or just caught).
- unit 2: saturated detector "ball is on row 8".

Layer 2 (four saturated units): R1 = [q_full ≥ 1 ∧ ¬row8], L1 = [q_full ≤ −1 ∧ ¬row8], R2 = [q_plain ≥ 1 ∧ row8], L2 = [q_plain ≤ −1 ∧ row8] (gain 500 on the linear units, threshold at q = 0.5, −200 gating from the row-8 detector).

Output: right = 20·(R1+R2)+20, left = 20·(L1+L2)+20, so the chosen move gets logit +20 and the other −20; no-op (and the equivalent up/down/fire) stay at 0 when neither fires.

```edits
# Unit 0: q_full = 2*bx - tx + [bx=0] - [bx=9] - px, scaled by 0.02
W1[3:600:60, 0] = 0.02
W1[9:600:60, 0] = 0.04
W1[15:600:60, 0] = 0.08
W1[21:600:60, 0] = 0.12
W1[27:600:60, 0] = 0.16
W1[33:600:60, 0] = 0.20
W1[39:600:60, 0] = 0.24
W1[45:600:60, 0] = 0.28
W1[51:600:60, 0] = 0.32
W1[57:600:60, 0] = 0.34
W1[10:600:60, 0] = -0.02
W1[16:600:60, 0] = -0.04
W1[22:600:60, 0] = -0.06
W1[28:600:60, 0] = -0.08
W1[34:600:60, 0] = -0.10
W1[40:600:60, 0] = -0.12
W1[46:600:60, 0] = -0.14
W1[52:600:60, 0] = -0.16
W1[58:600:60, 0] = -0.18
W1[540:600:6, 0] = [0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18]
# Unit 1: q_plain = bx - px, scaled by 0.02
W1[9:600:60, 1] = 0.02
W1[15:600:60, 1] = 0.04
W1[21:600:60, 1] = 0.06
W1[27:600:60, 1] = 0.08
W1[33:600:60, 1] = 0.10
W1[39:600:60, 1] = 0.12
W1[45:600:60, 1] = 0.14
W1[51:600:60, 1] = 0.16
W1[57:600:60, 1] = 0.18
W1[540:600:6, 1] = [0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18]
# Unit 2: ball on row 8 detector
W1[483:543:6, 2] = 12
b1[2] = -6
# Layer 2: R1, L1 (not row 8, lead target); R2, L2 (row 8, plain target)
W2[0, 0] = 500
W2[2, 0] = -100
b2[0] = -105
W2[0, 1] = -500
W2[2, 1] = -100
b2[1] = -105
W2[1, 2] = 500
W2[2, 2] = 100
b2[2] = -105
W2[1, 3] = -500
W2[2, 3] = 100
b2[3] = -105
# Output: right = action 3, left = action 1
W3[0, 3] = 20
W3[2, 3] = 20
b3[3] = 20
W3[1, 1] = 20
W3[3, 1] = 20
b3[1] = 20
```