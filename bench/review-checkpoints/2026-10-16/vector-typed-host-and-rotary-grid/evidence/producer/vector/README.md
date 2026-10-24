# Vector addition: typed host dispatch and source outcomes

This package adds five Lean modules without modifying the repository or the
existing execution families. It composes all 189 admitted canonical dtype pairs
with the actual existing grids: 81 integer/bool, 25 floating-five, 18 mixed-fp32,
36 mixed-fp16/bf16 and 29 fp64. The finite dispatcher is checked against the
existing 17×17 admission model and proves each selected route has exactly the
requested input signature. The other 100 pairs retain their first rejection:
64 at signature admission, including the 28 fnuz pairs whose standalone
promotion succeeds, and 36 at addition.

The launch primitive executes `IntegerSource.runGrid`, `TypedFloat.runGrid`,
`MixedFP32.runGrid`, `MixedSmall.runGrid` or `FP64Memory.runGrid`. Success/final
memory is derived from those proofs, never supplied as a launch oracle. The
original `HostWrapper.add` performs empty_like, CUDA assertion, numel, block-1024
grid launch and return in source order. Allocation and assertion failures are
retained; dtype-rejected launches append no returned-tensor event.

The carrier `HostWrapper.World (FP.Value fp32) (TypedHost.Native sigma)` retains
the existing shared state and adds the fp64 tagged memory in `native.wide`.
`project` selects shared cells for the 160 non-fp64 routes and distinct-width
tagged words for the 29 fp64 routes. Both memory carriers have explicit frame
theorems, the environment is unchanged by execution, and the unselected carrier
is unchanged. No equivalence between the carriers is silently assumed.

`EmptyLikeContract` is a concrete trusted primitive contract: output dtype,
shape, numel, CUDA/device metadata and zero storage offset; fresh output relative
to both inputs; preservation of every old cell outside the new output region.
Input preservation is proved from that contract. It does not falsely assert
that every empty_like output has the same strides as its input. SourceDomain
states the finite int32 length domain and sufficient y extent. Physical
pointer/view applicability, source-visible strides/offsets, no-wrap arithmetic
and unsafe/out-of-domain cases remain explicit work for acceptance.

The public success and exhaustive-outcome contracts require `ValueDomain`.
Shared inputs have exact tags and finite integer/float word encodings. For mixed
integer outputs, `DefinedConversions` preserves the inherited exact domain and
proves output independence from arbitrary undefined-value policies. The fp64
`Ready` premise retains typed active inputs, output/input separation and
`OutputDefined`; the latter is automatic for 20 floating-output routes and is
not removed for the nine integer/bool-output routes. The lower execution theorem
also preserves the old shared totalized semantics, explicitly parameterized by
undefined policies. Outside the wide defined domain, existing interpreter
faults remain mathematical diagnostics; the host diagnostic adapter's retained
state does not assert native rollback, traps or partial-store behavior.

The grids are the existing complete ordered launches. The fp64 execution chooses
the canonical complete permutation. Existing broader schedule results are
unchanged; this package does not add a GPU scheduling implementation theorem.
Numerical primitives and policies are inherited. The per-route binding proposal
in `evidence/numerical-bindings-proposal.json` binds actual statements, operations,
dtype pairs, callback/software ports and the new public contracts. It is not an
integrated queue entry or a native numerical validation result.

Public specifications in `TypedHostCorrect`:

- `vector_addition_typed_host_success`
- `vector_addition_typed_host_outcome`
- `vector_addition_typed_host_allocation_error`
- `vector_addition_typed_host_cuda_error`

Original-program acceptance remains pending; completion increment and new dtype
route increment are both zero. Remaining finite work includes pointer/view and
address-domain binding, outcomes outside the defined-value/safe-view domain,
trusted frontend/API target-domain binding, exact numerical queue integration,
schema2 source-stage/dtype/branch registration and independent substantive review.
Generic compiler/runtime/allocator/RNG/harness implementation verification is
not added as an obligation. No native compilation or GPU execution was run.

All real Lean attempts, including failures, have immutable source snapshots,
commands, exit codes and hashed logs. Final dependencies bind transitive project
sources/objects, direct external objects, original source, inherited admission
and numerical evidence, and the existing toolchain. Verification is performed
from the repository working directory, with outputs only in this package:

```sh
python3 -B /tmp/veritile-standardization-20261015/vector-typed-host-wip/freeze.py before
PATH=/tmp/veritile-lean-toolchain/bin:$PATH LAKE_HOME=/tmp/veritile-lean-toolchain /tmp/veritile-lean-toolchain/bin/lake env python3 -B /tmp/veritile-standardization-20261015/vector-typed-host-wip/verify.py bench/tritonbench_g/vector_addition/Proofs/TypedHostDispatch.lean bench/tritonbench_g/vector_addition/Proofs/TypedHostExecution.lean bench/tritonbench_g/vector_addition/Proofs/TypedHostWrapper.lean bench/tritonbench_g/vector_addition/Proofs/TypedHostValues.lean bench/tritonbench_g/vector_addition/TypedHostCorrect.lean
python3 -B /tmp/veritile-standardization-20261015/vector-typed-host-wip/freeze.py after
python3 -B /tmp/veritile-standardization-20261015/vector-typed-host-wip/freeze.py finalize
```

Finalization refuses to replace an existing receipt. Reproduce frozen work in
a separate copy and retain the original evidence.
