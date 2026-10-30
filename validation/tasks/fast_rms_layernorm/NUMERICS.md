# FP32 interpretation and proof scope

The selected source has FP32 storage and accumulation throughout. X is explicitly
cast to FP32 after loading. Ordinary forward loads W in storage dtype (FP32),
casts the normalized value to that dtype, multiplies W and stores FP32. Gemma
casts W to FP32 and multiplies by `W+1`. Forward squares each padded X lane,
reduces, divides by N, adds the FP32 conversion of host epsilon, applies rsqrt and
stores the saved r in FP32. Backward casts G/X/W/r to FP32; it forms `g*v`, reduces
`g*v*x*r`, and computes `(r/N)*(N*g*v - x*r*reduction)` in source order. dW folds
rows in increasing order, with FP32 zero initialization, the ordinary double cast
of normalized X, and separate multiply/add (`enable_fp_fusion=False`). No cast,
epsilon, operation order or reduction padding is silently discarded.

The public FP endpoints use canonical little-endian four-byte decoding to logical
32-bit words. Those physical reads populate the shared typed kernel interpreter.
Runtime epsilon is a distinct `Term.epsilon` leaf and an externally initialized
source register. `EpsilonABI` decodes the actual finite Python binary64 word as
an exact signed rational and rounds once to FP32, preserving both signs of zero
and signed underflow. The launcher binding states that its actual argument is
that word; it never identifies epsilon by matching an ℝ literal. Lean checks
both zero signs, both underflow signs, one and a halfway ties-to-even case.
The runtime reference specializes definitionally to the historical literal
reference when supplied `literal h.eps` at its epsilon position.

Their successful result is proved equal to an independent scalar expression
graph, then encoded into each of its four output bytes. Saved r is a source output;
backward consumes its current stored word. Empty backward skips all launches and
returns exactly all-bits-zero dW without a scalar literal identity assumption.

`NativeGraph` expands each selected reduction into a finite scalar binary tree;
its leaves are a permutation of all source lanes plus counted neutral leaves.
Scalar arithmetic, division, rsqrt, casts, literals, integer conversion, negation
and FMA are primitive interpretation ports. They never receive a tensor, row,
reduction result callback or whole-kernel success equation. Compiler tree choice,
operand order and FMA contraction/negation are explicit structural model choices.
The compiler-facing shape erases tensor and epsilon payloads. `mapWords_shape`
proves that arbitrary runtime word changes preserve every selector input shape.
Host scalar ABI conversion, primitive instruction meanings and compiler choices
are model boundaries, not numerical rewrite laws awaiting experiments.

There is no assumed reassociation, commutation or split/FMA equality. All input
words, including signed zero, infinities and NaNs, are admitted by the structural
FP theorem. No additional finiteness restriction is imposed at unused operands.
The theorem does not equate native FP32 rsqrt with exact reciprocal square root.
The Real endpoint proves logical outputs in the source's Real interpretation,
using actual input-pointer read laws. It has no native output codec premise and
does not assert that finite FP32 bytes can encode arbitrary real results.
The independent Real derivative proof has the explicit condition that each row's
variance-plus-epsilon is positive.

The compiler binding and source pin retain the production AST and its structural
lowering boundary. CUDA runtime/compiler/hardware conformance has not been tested
in this environment. This is a limit of the model's external binding, not a missing
kernel proof or deferred whole-program numerical assumption. CPU AST/autograd
regressions corroborate the source repair and input contract but are not the FP
proof. Lean checks use only the standard propext, Quot.sound and Classical.choice
axioms.

The 2026-10-21 pair-entry review retains the single-source native results as
ordinary theorems in `NativeCorrect.lean`. The main implementation-pair result
remains `proof_connection_pending`: the historical integer-index candidate has
not been connected to the selected repaired source and complete schedules.
This missing program proof adds no numerical assumption or experiment obligation.
