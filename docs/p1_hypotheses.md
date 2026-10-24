# P1 — predictions, written before the sweep

Recorded in advance so results can confirm or refute something, rather than be
narrated after the fact. Proposal line: *every design choice is a hypothesis,
not a result.* Date written: 2026-09-21. **Do not edit after the first real run** —
add a dated postscript instead.

## Design

- **Factors:** split (seen / held_out) × arm (zero_shot / few_shot / procedural)
- **Categories:** 4 seen, 10 held-out (unbalanced; report per-category means so
  the four seen categories cannot dominate a pooled average)
- **Samples:** n per cell, ≥3 — sampling is nondeterministic even without a
  temperature parameter, and one draw per cell gives no variance estimate
- **Primary outcome:** recognizability of the *rendered* drawing (P3)
- **Secondary, judge-free:** format compliance, spells-name rate, degenerate
  rate, ink ratio, canvas coverage, character entropy

## Hypotheses

**H1 — Leakage.** Seen categories score higher than held-out. A large gap means
recall is doing the work; a small gap means the skill is transferable.
*Refuted if* held-out ≈ seen across all arms.

**H2 — Prompting interacts with the split.** Few-shot helps mostly on *format*
(cleaner parses, better canvas use) and transfers weakly to unseen shapes,
because exemplars teach presentation, not anatomy. Procedural helps
disproportionately on *held-out*, because an explicit part-and-extent plan is
the transferable part of the skill.
*This interaction is P1's most interesting possible result.*
*Refuted if* arm effects are flat across splits.

**H3 — Procedural costs more and may hurt the easy cases.** Planning inflates
output tokens and may over-constrain categories the model already draws well,
so procedural may *underperform* zero-shot on seen.

**H4 — Format compliance is not the same as quality.** Parse-clean rate and
recognizability will be only weakly correlated: a well-formed blank canvas
parses perfectly. Reported separately, never summed.

## Failure modes that invalidate a cell, not just lower its score

Counted and reported separately from quality scores:

- spells the category name instead of drawing it
- blank or near-solid canvas (`degenerate`)
- no code fence / overflowed the canvas (`parse_issues`)

## Kill criteria — when to stop and rethink rather than scale up

- **Floor effect:** if the pilot shows near-zero recognizability on *seen*
  categories, the model cannot do the task and no arm comparison is meaningful.
  Change model before spending on the full grid.
- **Ceiling effect:** if every arm scores near the top on held-out, the
  categories are too easy; add harder held-out classes.
- **Judge invalid:** if the judge scores scrambled controls near real drawings,
  its scores are not measuring shape (Wang et al.) and P1's primary outcome is
  unusable until P3 fixes it.
