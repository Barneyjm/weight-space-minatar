# Seaquest — Fable 5.1 reports (verbatim)

## One-shot

Deliverables are in `.../scratchpad/wsp/fable_seaquest/`: `write_weights.py` (run; writes `weights.npz`, H1=254, H2=210, all finite) plus `check_weights.py` (my self-check).

**Strategy (priority rule, from the source).** Enemies/bullets/divers only occupy rows 1–8 and never change row, and the player moves before them, so a vertical step is the universal escape. The net: (1) at the surface, dive unless row 1 is hazardous near me; (2) if a hazard sits in my row within ~3–4 cells (bullets) / ~2–3 (fish, subs), move up or down, never into a row that has a hazard near my column, never surface with 0 divers, never "down" at row 8 (no-op), prefer down at row 1 to avoid a diver-costing surface; (3) oxygen < 60 with ≥1 diver, or 6 divers aboard → go up; (4) diver in my row → chase it (vertical seeking suppressed), else move toward a diver above/below (adjacent row preferred); (5) enemy ahead in my row → fire; enemy behind → turn toward it; (6) drift off spawn columns 0/9; idle = stay.

**Layers.** Layer 1 (gain 20) is 254 exact binary ANDs on the one-hot board: player-in-(row, 2-col bin) × hazard-window in my row / row above / row below (40+35+40), enemy and diver left/right of me in 3 column bins (96), diver-in-my-row and divers above/below (34), facing (sub_x − back_x), surface/row-1/row-8/oxygen/diver-gauge flags. Layer 2 passes units through (gain 20) and adds four ANDs (enemy-side × facing → shoot/turn). Layer 3 is a logit table: dodge +50/suppress −100 > destination-row block −60 > ascend/dive ±30 > seek 14/13/10 > shoot 12 > idle 3.

**Weaknesses.** Coarse column bins (waits when a diver is inside my bin; dodges enemies I could shoot at distance 2); no fallback other than random vertical when both adjacent rows are hazardous; no awareness of bullet direction; difficulty ramp eventually fills rows so death at ramp ~10–15 is likely; sticky actions can double-step into an unchecked row.

**Predicted mean score:** ~8–15 (kills dominate, occasional 6-diver cash-ins).

**Checks run:** no game code executed, no rollouts. Built 30 synthetic 10×10×10 boards by hand (surface, dodge cases at rows 1/4/8, blocked rows, divers in each relation, low oxygen, 6 divers, edges, facing/turn cases) and compared the net's logits to an independent plain-Python reference of the rule: all match to 5e-8; min |layer-1 pre-activation| = 10. This check caught one real bug (a bin window misattributing a diver's side), which I fixed.

## Revision 1

Revision 1 saved: `fable_seaquest/weights_v2.npz` (H1=256, H2=228) built by `fable_seaquest/write_weights_v2.py`; `weights.npz` untouched. Self-check in `check_weights_v2.py`: 27 hand-built boards (including re-creations of the failure frames) vs an independent reference of the rule, all logits match to 5e-8, every layer-1 unit saturated.

What changed and why (from `failures_v1.txt`):
- Sticky-repeat kills (66% of deaths): side-specific danger units HL/HR (hazard on my left/right within ~3-4 cells, bullets ~4-5) replace D. They trigger the vertical dodge and also block any horizontal step toward the hazard (-100), so diver-chasing and turning can no longer walk up to an enemy where one repeated `r` is fatal. If a hazard is on one side and both vertical rows are blocked, flee away horizontally.
- Row-above/below blocks now cover only fatal entries (enemy <=1-2, bullet <=2-3) but look two rows ahead (sticky double step) and weigh -150, above any sum of positive drives (v1's -60 lost to dodge+ascend and even to idle noise).
- Idle is near-deterministic (n=6, others 0) and row 2 with no divers blocks `u` unless oxygen is low: ends the "random `u` then sticky `u` surfaces diver-less" deaths.
- Oxygen: escalating ascend (+30/+40/+100 at <60/<40/<20, the last overrides blocks) plus a small horizontal walk while blocked.
- Edge columns: +10 to leave x=0/9, -10 to step onto them (spawn-on-top deaths).
- Paid for it with 2-half (instead of 3-bin) enemy/diver side units; divers always move toward the centre so this costs little.

Predicted mean score: ~30 (longer lives; same kill/diver logic).
