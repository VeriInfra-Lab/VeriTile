# Checked token_attn_reduceV

`Kernels.lean` exposes the existing source-only `Source.lean`; `Correct.lean`
and `FPEquiv.lean` are the independent public proof entries.

This is a gathered weighted value sum. It mutates every element of the supplied
`Out[B,Hq,D]`, returns `None`, and preserves every other physical cell. A
zero-length batch row is written as zero. There is no softmax calculation.

For each batch, the independent Real target is the weighted sum of `P[h,start+n] * V[table[request,n],h/(Hq/Hkv),d]` over `n<sequence`. The same sequence limit governs P and V loads; masked V lanes do not read unused cache rows.

The selected contract is contiguous FP32 P/V/Out on one CUDA device, signed-int32
request tables and batch vectors, D in {16,32,64,128}, positive divisible heads,
and no lazy-negative views. Guards check shapes, metadata bounds, active cache
IDs, padded signed-int32 address expressions, four-byte nonempty alignment,
pointer backing ends and output/read byte interval separation. Empty batch
returns without launch, alias or alignment checks. Canonical strides are derived
from shapes, including degenerate empty tensors. Only integer metadata is
inspected on the host. Supplied tensors and output storage are retained directly.

The kernel uses 128-token tiles, four warps and one stage. The historical
probability-step correction `start_n * stride_pbs` is retained. The independent
public `Host.correct` and `Host.fpequiv` roots reuse complete typed AST body, loop,
grid and store proofs. Checked admission derives active address safety, padded
expressions (including the pre-loop offsets), metadata interpretation, finite
word placement and a complete physical byte frame. Validation distinguishes
empty and launching calls. The mathematical target erases rounding; the FP
target retains every masked product, reduction node, loop addition and output
cast. Ordered reduction trees and allowed contraction schedules are structural
parameters. The only numerical ports are rounded scalar FP32 add/multiply/FMA;
there is no whole-dot, whole-reduction or whole-output numerical oracle.

Tensor storage interpretation and successful CUDA execution are ordinary runtime
boundaries. Shared launcher grid-product compatibility is deferred in
`bench/TODO.md`. Historical pinned SM80 compiler profiles keep their restricted
specialization domains; they do not certify arbitrary JIT specializations.

Validation: all 20 Lean modules passed, including axiom and target
non-circularity audits. `python bench/repairs/tritonbench_g/token_attn_reduceV/check_contract.py`
checks 48 CPU interpreter/guard cases across every D, tile tails, GQA, empty
batches/caches, input frames, invalid metadata, negative views, overlap and
unaligned storage. `check_lowering.py` checks eight offline SM80 compilations,
including scalar-one specialization. No GPU numerical conformance was measured.

The checked public `checked_call` / `checked_fp_call` endpoints compose empty and launch validation. Empty dispatch preserves the supplied descriptor and current memory without canonical inactive strides. Live calls reject mixed FP32/int32 input-byte overlap while retaining same-dtype input aliases. The regression checker uses a fixed random seed; its comparison against a double-precision reference is an approximation check, separate from the word-level FP theorem.
