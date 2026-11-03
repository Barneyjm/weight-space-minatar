# Opus 5.5 — design notes and self-reported mistakes

Opus wrote its networks in the main session under the same rules:
- It read only the MinAtar source.
- It never ran the game or the network on game states before scoring.
- It used no training, fitting or search.
- Its only checks were synthetic states compared against its own reference rule.

## Breakout
- **One shot: 7.1.** Opus planned with single-step "fold" arithmetic and left the paddle idle on the frame after a bounce. The paddle moves at the same speed as the ball, so wasting that frame meant missing by one column.
- **Revision 1: 23.0.** It looked at three traces of lost balls, gave bounce frames a target, and fixed a gate bug it found by reading the code. Breakout had no formal revision protocol yet, so this revision is informal.

## Asterix
- **One shot: 13.0.** The network has one-step safety per target cell:

  `s = 2E(c) + E(c-1) + E(c+1) - T(c)`

  There is one entity per row, so a trail on c means the neighbor is moving away. It also seeks gold and keeps off the edges.
- **Weakness:** it only looks one step ahead and random-walks among safe moves, so it walks into spots with no exit.

## Seaquest
- **One shot: 13.2.** Same safety idea, with enemy bullets counted on either side. It fires at enemies ahead, turns toward enemies behind, and surfaces when oxygen is under 60 if it has a diver.
  - Known blind spot: a diver's trail can cancel an enemy threat. This affected 0.7% of synthetic states.
- **Revision 1: 10.5, worse.** About 85% of deaths were sticky repeats. Opus answered by making the sub stay still and removing turning.
  - Survival rose from 426 to 564 steps, but kills fell.
  - It optimized against causes of death without checking where points came from.
