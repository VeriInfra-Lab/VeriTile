---
title: Conventions
description: TileShape orientation, ND-general operators, naming patterns, and where to push back on shortcuts.
---

The conventions below are not stylistic preferences — each has bitten us
hard enough to be a project rule.

## Outermost-first shapes

`TileShape := List Nat` is **outermost-first**:
`shape (M, N) = [M, N]`. The list index *is* the user-visible axis index,
matching Triton, NumPy, PyTorch.

When implementing a new axis-aware operator, the natural Lean move is to
align the head of the list with the "active" axis so structural recursion
on the head is easy. For broadcasting and head-axis reduction,
innermost-first internal storage feels cleaner.

Don't do it.

**Why:** users paste real `.py` kernels and write `axis=K` referring to
the user-visible outer-first shape. Internal flipping is a permanent
ergonomic tax — every DSL macro and every doc example would need
translation in the user's head. Triton semantic fidelity wins.

**How to apply:** when a new ND operator (reduce, transpose, expand_dims,
ND-broadcast extension, …) feels easier with innermost-first, eat the
Lean-side pain on the outermost-first side:

- `init ++ [last]` recursion patterns.
- Structural induction on `init`.
- Helper `_pos` (positivity-asserting) variants to avoid `Option`.

The reduce-axis implementation in `VeriTile/Triton/Semantics/TileOps.lean` —
`Tile.reduceSumDrop`, `Tile.reduceMaxDrop` — is the canonical
template.

## Always go fully ND

Before writing `match shape with | [_, _] => ...`, or `(rest ++ [axisDim])`,
or `Input2DAt`: ask whether a future kernel could want a different rank or
axis here. The answer is almost always yes.

The discipline:

1. **Design with `Fin shape.length` axis indices.** New axis-aware
   typed-AST nodes take an explicit axis parameter, not a hard-coded
   position.
2. **Parameterize ND infrastructure by offset functions.** Predicates,
   helpers, and theorems take `TileIndex shape → Nat` rather than baking
   in 1D / 2D patterns.
3. **Provide named offset families alongside.** `Offset.linear1D`,
   `Offset.rowMajor2D`, `Offset.contig` are ergonomic helpers; the core
   API stays generic.

The reduce-axis API in `VeriTile/Triton/Core/Shape.lean` — `axisDim`,
`eraseAxis`, `setAxisOne`, `reduceShape`, `insertAxisIndex`,
`replaceAxisIndex` — is the canonical template. Mirror that pattern for
any new axis-aware op.

**What this costs:** real dependent-type work, but bounded. You pay it
once. The alternative (rank-/axis-specific helpers) costs a future rewrite
each time the bench picks up a new shape pattern.

## Rewriting cost is a tiebreaker, not a thesis

When evaluating design tradeoffs:

- Treat "but this would mean rewriting N existing proofs" as a tactical
  concern (when to ship, how to slice the work), **not** a strategic
  vote.
- If two designs are otherwise close, the one that doesn't require
  rewriting may win by a hair. If one is meaningfully cleaner, more
  correct, or more Triton-faithful, recommend it even when it implies
  a multi-hundred-line proof rewrite.
- Don't recommend a design whose **primary** justification is "preserves
  existing proofs". That's a tiebreaker, not a thesis.
- Tactical ship-timing overrides are legitimate ("let's do D-b now and
  revisit D-a after the release"). But the recommendation should not
  arrive at D-b by default.

## Naming patterns

A handful of patterns recur across the bench corpus; staying consistent
makes proofs cheaper.

- **Per-kernel files** live at `bench/tritonbench_g/<kernel>/<Kernel>.lean`
  for ports, `bench/examples/<Kernel>.lean` for examples.
- **Theorem names** end in `_correct` (kernel ↔ spec) or `_refine`
  (kernel ↔ kernel). Compute-layer projections get a parallel
  `_compute_correct` / `_compute_refine`.
- **Spec definitions** typically named `<kernel>_spec` or `<feature>Spec`.
- **Offset / index helpers** name the indexing shape:
  `inOffsetDef`, `outOffsetDef`, `outOffsetFn`, etc.
- **`_pos` suffix** marks the positivity-asserting variant of a helper —
  returns the underlying value directly instead of `Option`.

Bench headlines use `specification`: `Spec.Real (io ⊨ formula)` for
mathematical correctness and `original ≡[R] candidate` for FP equivalence.
Historical rounding-model IO notation retains its original meaning. Naming suffixes are not how the audit discovers
headlines; follow the [main-theorem conventions](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/MAIN_THEOREM_CONVENTIONS.md).

## Antiquotes are paste-in-hostile — minimize them

`$(REGION)` in `tl.load($(xReg) + offs)` and `$ℝ(t)` for ℝ literals are
both VeriTile inventions that Triton's `.py` doesn't have. They're a
friction point for the paste-in goal.

Bind input types, constexpr values and imported helpers outside the copied
`triton { ... }` function body, as described in the
[translation guide](/VeriTile/cookbook/translating/). The legacy antiquote forms
remain useful when reading older library examples or writing AST proof fixtures.
New source transcriptions retain Python spelling and do not insert them.

Known boundaries still need explicit checks:

- `tl.toReal(_)` is a VeriTile-specific lower-level conversion; it must not be
  inserted into a copied Python body.
- Integer channels use mathematical `Nat`/`Int`; fixed-width behavior requires
  the relevant model or range conditions.
- Memory kwargs have a whitelist. Literal `eviction_policy="evict_first"` and
  `"evict_last"` are supported metadata; unknown kwargs such as an unsupported
  `cache_modifier` still fail rather than silently changing the body.

These are recorded tradeoffs, not surprises. New constructs should not
add to the list.
