# Checked context_attn_mistral

Reference responsibilities (2026-10-23): The source decomposition, ordered accumulator recurrence and execution invariant are in `KernelOnly/Proofs/Reference.lean`; `KernelOnly/Reference.lean` is an import-only compatibility entry. The complete main FP comparison still uses this recurrence through the native array implementation.

The checked wrapper mutates the supplied packed `O[T,Hq,D]`, returns `None`,
and launches the original single grid. It retains FP32 Q/K/V/O, D in
{16,32,64,128}, positive divisible grouped heads, 128-by-128 attention tiles,
four warps through D64 and eight warps for D128, and one stage. All tensors
are contiguous on one CUDA device without lazy-negative views. Only signed
integer metadata is inspected on the host; floating payloads and output are
never resolved, cloned, copied or copied back.

Starts and lengths must be nonnegative int32 values, bounded by T, with
pairwise disjoint nonempty packed intervals. Live launches check the CUDA grid
axes, signed-i64 dense stride arguments, complete pointer backing bounds,
four-byte nonempty alignment, and output/read interval separation. Read views
may alias each other, including mixed-dtype raw-word aliases. Empty batch or
zero block count returns before launch-only checks. An arbitrary Python window
is clamped to the equivalent nonnegative signed-i64 range. `max_input_len` may
underlaunch: only rows below `ceil(max_input_len/128)*128` are written.

The independent mathematical result preserves the source's finite -1e9
sentinels. A query row visits `(row/128+1)*128` key positions. Causal/window
failures have score -1e9, padded values are zero, and every visited sentinel
weight remains in the denominator. This is not the ideal hard-mask formula.
For an all-masked window, the mathematical sentinel weights are positive;
the native FP32 exponential underflows to zero and the guarded native result
is zero. The two independent theorems make no numerical-accuracy claim between
these interpretations.

`WholeReal.checked_call` proves the mathematical result using complete actual
Real AST body execution and then its projected physical stores. The ordinary
`Mathematical.Adapter` binds current input coordinates and signed metadata to
source roles; none of its fields mentions execution or results. Logical role
reification is a proof interpretation and creates no Python allocation.

`WholeFP.checked_call` consumes `SourceGraph.Binding` for the canonical
`Program.kernel` and proves the selected scalar/store graph against an
independent initial-word online recurrence. It does not claim a full AST
register-value invariant for FP. The graph includes both finite masks, both
floating guards, actual marshalled scale, natural-exp lowering to exp2,
ordered physical reduction trees and routes, normalizer FMA, scaled live
accumulator, both TF32 WGMMA chains, the P-to-V cast and output cast. Every
k8 instruction has an explicit clear/live selector: QK starts cleared; PV
retains its live accumulator even if that word is zero. Numerical obligations
are local scalar FP32 and fixed m64n{16,32,64,128}k8 WGMMA instruction semantics.
There is no whole-dot, whole-reduction, whole-tensor or execution-result oracle.

Both endpoints compose checked empty/launch dispatch, preserve the exact
supplied output descriptor, and return None. Dynamic stores read current
physical memory; separation proves the reads stay equal to initial inputs.
Packed coordinate injectivity proves readback for every store. The byte frame
covers the precise complement of stores, including packed gaps, underlaunched
rows, every input/metadata view and unrelated arena bytes. The FP endpoint also proves all four output bytes using the ordinary local
little-endian raw-word codec law. Native pointer,
mask, loop and window correspondence follows from checked geometry and
metadata; complete four-byte word bounds and source-pointer equality are proved.

Compiler correspondence is an external structural boundary for the selected
specialization: `SourceGraph.Binding` binds the actual parsed source fragments
to selected raw input leaves, initialization, one iteration, loop control,
final cast and store operands. `selectedDispatch` executes these supplied
fragments and the proof consumes their local maps. The fragment code indices
and `source_split` pin the source; they do not establish compiler correspondence
by themselves. The actual specialization's trees, routes, owners and instruction
graph must match. The eight saved profiles do not establish this universally.
The historical native profile helpers remain available with their original
scope. Offline compilation checks the current source's selected instruction
families and clear/live selectors; it does not prove an entire compiler or
all register routes. Tensor/read/write interpretation, metadata observation,
no concurrent mutation, and successful CUDA execution are ordinary runtime
boundaries. Shared launcher grid-product compatibility remains deferred in
`bench/TODO.md`.

Validation: 26 Lean modules, axiom and target non-circularity audits; 48 CPU
interpreter/guard/source-mapping cases; eight offline SM90 compilations across
all D and scalar-one specialization. Reproduce the CPU checks with
`python bench/repairs/tritonbench_g/context_attn_mistral/check_contract.py`
and the compiler checks with the adjacent `check_lowering.py`. No GPU
numerical conformance was measured.
