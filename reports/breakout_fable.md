# Breakout — Fable 5.1 one-shot report (verbatim)

## Deliverables
- `fable/write_weights.py` (run; writes `fable/weights.npz`, H1=237, H2=20). Also `fable/check_synthetic.py` (my sanity check).

## Strategy
Predict the paddle cell where the ball will be on row 8 at the paddle check and move there, then hold. From the source: the ball's x path is independent of vertical bounces, X(t)=tri(bx+dx·t) with tri the period-20 triangle (0..9,9..0, wall stall included). Only the x-step count depends on vertical geometry: moving down t=8−by; moving up t=by+7−2r where r is the first brick row hit on the diagonal (3, 2, 1, or −1 = top wall). Catch condition is pos ∈ {X(k−1), X(k)}; I target X(k−1).

## Layers
- L1 (237 tanh units, gain 20, exact 0/1 inputs ⇒ saturated): paddle thermometer (pos≥j, 9 units); direction signs dx/dy from ball−trail with wall/bounce fix-ups (trail==ball column at x=0 ⇒ right; trail row==ball row ⇒ down unless row 8); thermometer staircases of tri(u) for u=bx±(8−by) (down) and u=bx±(by+o), o∈{1,3,5,9} (up cases r=3,2,1,wall); brick lookups along the ball's diagonal (ball-on-diagonal AND brick at (r, x_r) AND no lower brick), 90 units.
- L2 (20 units): for each (direction, case) a "pos>T" and "pos<T" unit, gate −200 if direction/case mismatch; T read linearly from the staircase.
- L3: left=+15/−15, right likewise, noop 0, u/d/f −30 ⇒ wrong-action sampling prob ≈3e-7.

## Weaknesses
Ball rattling inside the brick region (bounces from above, pass-through via `strike`) mispredicts for a few steps; sticky actions can cost a step when the target is far (worst case 8 cells in 9 steps). Trail-row==ball-row at rows<8 is assumed "moving down".

## Predicted mean score
Mostly limited by rare sticky/rattle misses; guess 60–150 (10k-step cap ~ few hundred max).

## Checks run (synthetic only, no game code executed)
Forward pass on 40,000 hand-built obs arrays (all ball cells × valid trail offsets × 10 paddle positions × 8 brick maps) compared to a plain-Python reference of my intended rule: 0 mismatches, min logit margin 15; repeated in float32; six hand-walked trajectories (e.g. reset states, post-paddle-bounce) verified T by hand.
