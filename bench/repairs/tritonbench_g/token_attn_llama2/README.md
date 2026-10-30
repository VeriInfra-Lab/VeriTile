# Checked token_attn_llama2

This is the gathered QK score stage. It mutates `Out[Hq,T]`, returns `None`,
and preserves every other physical cell. There is no softmax or value reduction.
For `t < Length[b]`, the independent Real target is

```
Out[h, Start[b]+t] = sum_d Q[b,h,d] * K[Loc[b,M-Length[b]+t], h/(Hq/Hkv), d] / sqrt(D).
```

The selected contract is contiguous FP32 Q/K/Out on one CUDA device, int32
Loc/Start/Length, D in {16,32,64,128}, positive divisible head counts, and no
lazy-negative views. Nonempty launches require canonical dense strides,
four-byte aligned nonempty views, disjoint output/input byte intervals,
nonoverlapping active output segments, valid active suffix page IDs, and the
checked padded int32 envelope. Shapes and all metadata values are checked.
Only integer metadata is inspected on the host; no floating payload or output
is cloned. Standard empty layouts return without launching or alignment checks.

The original floating kernel is AST-identical, including 32-token tiles,
source multiply/reduction/scale order, two warps and one stage. Public
`Whole.correct` and `Whole.checked_fpequiv` have independent import roots.
They reuse full typed AST execution, arbitrary complete grid schedules, every
logical output and the full current-storage frame. `Proofs/Checked` derives
metadata injectivity, finite placement, complete physical words and byte frames
from guards, and models empty versus launch validation separately. Tensor
metadata/storage interpretation and successful device/runtime execution remain
ordinary library boundaries. Shared launcher overflow compatibility is deferred
as recorded in bench/TODO.md.

The FP target is an independent initial-word expression graph with complete
ordered dimension trees, allowed source FMA contractions and the actual
marshalled scale bits. The only numerical ports are rounded FP32 add, multiply
and FMA. They do not assert exact Real arithmetic. Tree and contraction choices
are structural compiler parameters, not scalar result callbacks. Historical
pinned SM80 profiles and `CompilerBinding` remain in helper modules with their
original restricted specialization domains; arbitrary JIT specializations are
not thereby certified as those profiles.

Validation: all 24 Lean modules in the public closure passed, including axiom
and non-circular target audits. 51 CPU interpreter/guard cases passed (all four
D values, lengths 0/1/31/32/33/65, empty batch/cache, suffix metadata, GQA,
frames, negative views, invalid metadata, overlap and unaligned storage).
Eight current-source offline SM80 compilations passed, including scalar-one
specialization. These are structural and CPU checks; no GPU numerical
conformance has been measured.

The checked public `checked_call` / `checked_fp_call` endpoints compose empty and launch validation. Empty dispatch preserves the supplied descriptor and current memory without canonical inactive strides. Live calls reject mixed FP32/int32 input-byte overlap while retaining same-dtype input aliases. The regression checker uses a fixed random seed; its comparison against a double-precision reference is an approximation check, separate from the word-level FP theorem.
