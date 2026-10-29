# Checked FP32 SwiGLU forward and backward kernels

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

The original benchmark explicitly creates FP32 A, B and DC. Both JIT bodies,
the `silu` helper and every `calculate_settings` branch remain AST-identical.
`Kernels.lean` directly quotes the selected FP32 source with actual contiguous
row stride `cols`, including non-power-of-two widths. There is no handwritten
replacement AST or new arithmetic candidate.

The independent mathematical formulas are, with `s=1/(1+exp(-a))`:

```
c  = a*s*b
da = dc*b*s*(1+a*(1-s))
db = dc*a*s
```

The retained derivative proofs connect DA/DB to the forward formula.
`swiglu_kernel_correct` proves successful execution of the complete
selected forward/backward kernel, every active output and the full exterior
regional memory frame. Both backward values use the entry-state A/B/DC.
`swiglu_tensor_kernel_correct` composes this result with one physical reader
and the actual admitted Tensor pointer/offset binding.

The wrapper provides assertions: contiguous strided FP32 CUDA payloads without
lazy negative bits, matching devices and element counts, positive final width,
bounded row/index arithmetic, aligned nonempty pointers and pairwise disjoint
live input spans. Forward shapes match; backward accepts the flattened A/B
views returned by forward. Same-storage disjoint slices are allowed. The
existing conservative forward A/B separation remains explicit; relaxing this
read-only restriction is an input-contract TODO, not a numerical assumption.

`InputBinding.InputContract` includes both the payload checks and the unchanged
`calculate_settings` limit `blockSize(cols) <= 65536`; `checkInputs_iff` connects
the combined check to this contract. Existing `pointer`, `indices` and `mask`
proofs supply four-byte alignment, exact active addresses and finite-width
index/mask correspondence. Empty batches query no pointers and launch no kernel.
Forward allocates C; backward writes caller A/B in place. There are no input
copies, padding buffers, snapshots for execution, or copyback.

`Misc/NativeCorrect.lean`’s ordinary `swiglu_kernel_native` theorem proves full successful source graph execution,
all active native outputs and the exterior typed-memory frame. Its independent
native reference preserves the actual instruction order:

```
s    = div.full(1, add(exp2.approx(mul(sub(+0,a), encoded_log2e)), 1))
silu = mul(s,a)
c    = mul(silu,b)
db   = mul(silu,dc)
da   = mul(mul(fma(silu,sub(1,s),s),dc),b)
```

The sigmoid constants are encoded FP32 words; the backward FMA is retained.
Loads enter as raw word leaves. No finite-input condition, primitive equality,
FMA-versus-split law or output premise is needed. **Numerical transformation
assumptions: none.** Sigmoid/exp2/division/FMA realization remains the shared
instruction/compiler modeling boundary, not deferred numerical experiments.
Real is the independent mathematical interpretation, not native exactness.

The prior full-host and optional primitive-conformance transports are preserved
in `Proofs/LegacyCorrect.lean` and `Proofs/LegacyFPEquiv.lean`. The historical
padded-layout helpers remain available; the public kernel uses `directSource`.
No wrapper, allocator or runtime implementation proof was added.

`check_contract.py` checks unchanged JIT/settings ASTs and CPU physical-storage
output/frame behavior for forward and in-place backward, offsets, empty
batches, width 65536, invalid metadata and CUDA/HIP configuration branches.
`check_cuda_lowering.py` additionally checks SM90 PTX register dependencies,
encoded constants, FMA placement, stores and FP32 memory for seven configurations
of each kernel and all six related `swiglu_fwd` block sizes. CUDA is mocked in
CPU replay; offline compilation performs no GPU execution. These are source and
lowering checks, not independent numerical replay.

Pointer/storage observations follow the shared Torch
[`data_ptr`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
[`storage_offset`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html)
and [`is_contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html)
contracts. FP32 natural alignment follows
[PTX addresses as operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).
The common runtime grid-product/segmented-launch work remains deferred.

Concrete original-source counterexamples are retained in `check_contract.py`: The original single-stride launch reads row padding when B has row stride 4 but A/C have width 3. Aliasing backward A and B overwrites the distinct DA result with DB. The checked contract excludes these cases explicitly; the unchecked original is not retroactively certified.

`Misc/TensorBridge.lean` contains `TensorBridge.swiglu_tensor_kernel_correct` and
the optional Python-check/native-pointer bindings as ordinary theorems. The
Real specification and ordinary native theorem keep their existing names and scope.

## Public IO format (2026-10-20)

`Correct.swigluIO` and `FPEquiv.swigluNativeIO` declare the actual source,
three logical input channels, three possible output channels, the contiguous
`row * cols + lane` windows, and branch-specific active masks. Forward uses
A/B and writes C; backward uses A/B/DC and writes A/B in place. Forward places
no binding or capacity requirement on DC. There is no private scratch.

The Real result and auxiliary native result use the shared `FP.Structural.ProgramIO.Implements`
judgment. It proves the actual source's load/store trace safety, successful
execution, each enabled independent logical result, and the exact raw typed-cell
frame. `cols ≤ block` retains coverage of every column. The source precision
remains FP32; `.real` labels the Structural interpreter's cell channel, not a
change to the Python or native precision.

`Proofs/SourceSafety.lean` checks the actual statements in their evaluated
states. `Proofs/LogicalExecution.lean` reuses the existing source execution and
readback proofs for active logical inputs. `Proofs/PublicIO.lean` connects those
facts to the shared IO judgment. The optional Tensor theorem retains its full
statement and uses the same logical execution foundation. No Python, kernel,
input-contract or numerical domain has changed. This regional IO result does
not itself claim a new flat-pointer relocation theorem; the existing physical
binding helpers remain separate.

The auxiliary native result compares the single selected source with its independent
ordered sigmoid/FMA reference, admitting every FP32 word. It introduces no
floating rewrite or numerical experiment. Targeted builds of both public
entries, all new helpers, and `Misc/TensorBridge.lean` passed with standard base
axioms. No GPU execution or independent numerical replay was run in this pass.

## Actual source pairs (2026-10-21)

`FPEquiv.forward_index_equiv` and `FPEquiv.backward_index_equiv` are the main
`original ≡[[]] indexCommuted` specifications. They reuse the existing typed
source variants at the repair's FP32 storage and contiguous stride `cols`.
The candidate exchanges only the signed integer row-base multiplication;
floating operations, conversions, masks, backward FMA and ordered A/B stores
are unchanged. `IO.lean` names both actual complete sources, and
`Proofs/IndexEquivalence.lean` reuses the existing signed prefix theorem.
The pair preserves complete state and failure behavior for every initial row
and arbitrary floating backend, without numerical assumptions. The successful
logical IO and word-observation theorem remains in `Misc/NativeCorrect.lean`, with
the original declaration name available by separately importing that optional entry.

## Source readability cleanup (2026-10-22)

`Kernels.lean` now directly contains the four complete ordinary `triton`
quotations for the selected FP32 forward/backward original and index-commuted
kernels. The explicit input/output metadata around each full quotation matches
the historical source. `Proofs/SourcePreservation.lean` proves complete function
equality by `rfl` with each existing typed-source specialization. Named original
interfaces, actual contiguous and historical padded strides, casts, silu/sigmoid,
FMA distinctions and ordered in-place stores are preserved. The original generic
API and its other historical dtype profiles remain unchanged compatibility
sources; they do not generate the selected repair's authoritative programs.

The scoped source-only delivery records targeted dependency builds and a fresh
joint check of the existing public and optional entries. This organizational
change does not reopen proof completion, change contracts or add a numerical,
host, compiler or hardware acceptance requirement.

`IO.lean` also contains the canonical `PublicIO` channels, read/write maps,
masks and single-source schedule for both forward and backward.
`Correct.lean` selects its Real interpretation without repeating the record;
`Proofs/PublicIO.lean` keeps the source execution, safety and frame proof.
