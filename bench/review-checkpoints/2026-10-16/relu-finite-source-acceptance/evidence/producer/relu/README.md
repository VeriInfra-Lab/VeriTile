ReLU finite source acceptance candidate
======================================

This additive package proposes complete finite source-scope coverage for the
original `relu_triton_kernel.py`. It does not mark the repository accepted.
Independent root review, integration of the exact numerical authority, current
catalog regeneration and the unchanged scope checker remain required.

Copy the five `.lean` files to the same repository-relative paths. The public
statement is `FiniteCorrect.relu_finite_source_outcomes`. It covers all seventeen
canonical source-target profiles: twelve existing typed routes, fp64, two fp8
ordered-comparison failures and two unsupported fp8 signatures. Device,
allocation, numel and launch-setup errors preserve the first error and source
event prefix. Success executes the existing source grid; no launch success or
final-value equation is a premise. The newly added modules all include the
standard axiom audit and use no additional axioms.

The finite API is actual Torch strided CUDA tensors, canonical dtypes from the
frozen inventory, CUDA sm90/Triton3.7.1/Torch2.13.0+cu130 and the source's fixed
block1024. `0 <= N <= 2^31` covers empty tensors, partial blocks and multiple
blocks. CUDA device strings include indexed devices. The frontend evidence now
checks all seventeen profiles at N=0,1,1024,1025,2^31-1:65 frontend successes and
twenty expected rejections. Generalizing the type classification to the interval
uses the trusted frontend type primitive and source inspection: N affects only
the mask bound (signed i32 below2^31, unsigned i32 at2^31), while tile shape, pointer types, value comparison/where promotion and store conversion stay fixed. These probes are not an exhaustive native proof. An additional68-pair boundary audit covers N=2^31,2^31+1,2^32-1,2^32: all thirteen value routes still frontend-generate. Exactly N=2^31 is safe and included by FiniteIndices; the next block at larger N overflows the signed source multiply before the gate. This is an explicit unsafe configuration, not a frontend rejection.

The trusted boundary is explicit. `TensorSemantics.view` denotes the actual
storage base, capacity and byte permissions. Its `emptyLikeStrides` denotes the
actual Torch default preserve_format result, including non-dense inputs; it has
no access to kernel values or outcomes. `PrimitivesContract` specifies read-only
device/numel queries and successful empty_like's exact metadata, fresh region,
physical separation, valid writable output and preservation of prior non-output
cells. Allocation contents remain arbitrary. Typed input views bind actual
storage addresses to finite integer/bool words or correctly tagged float values;
fp64 uses complete64-bit words. `access_summary` derives active finite addresses,
permissions and input/output disjointness, including nonzero storage offsets.
These are primitive contracts, not allocator/compiler implementation proofs.

The result is in consecutive physical input/output order. The kernel ignores
strides, so safe transposed and sliced views are covered as written without a
false logical elementwise claim. An expanded view whose numel exceeds its flat
storage extent is explicitly unsafe. Missing permissions, pointer wrap and
N above2^31 are not silently converted to frontend errors
or waived numerically. No source assertion enforcing these conditions is added.
Complex/quantized/packed/shell dtypes, arbitrary Python objects, CPU and other
targets are outside the reviewed canonical CUDA API; every previously admitted
canonical dtype route is retained. See `evidence/domain-review.json`.

Only pid0 stores. Every other launched pid still performs the pre-gate source
load, comparison and selection. Success writes exactly indices `<min(N,1024)`;
the entire complement is framed, including the N=0 case. Output indices>=1024
retain arbitrary allocation contents. This package never asserts whole-tensor
mathematical ReLU for N>1024. The old typed launch uses the established increasing
pid source schedule; the fp64 route supports every complete program permutation.
Fresh disjoint input/output views and the sole writing pid rule out cross-program
write races under the trusted Triton load/store/launch semantics. No native
instruction-interleaving, compiler or GPU implementation proof is claimed.

The source inventory retains the non-JIT `test_relu` helper and its import-time
test initialization. It has no production caller, so the proposal marks it
test_harness. Its four calls already use the modeled source wrapper. No RNG or
general module implementation proof is introduced.

Numerical and schema2 proposals
------------------------------

`propose.py` checks the exact original host/kernel AST and frontend records and
generates `evidence/declaration-proposal.json`, `scope-row-proposal.json`,
`scope-cases.json` and `numerical-authority-delta.json`. The340 cases are17 dtypes
times4 disjoint N ranges times5 first/terminal outcomes. The rule delta has12
families and84 exact declaration/configuration/port bindings on39 nonempty
successful cases. Empty launches and error cases require no numerical callback.
Comparison, zeros, extension and fp64 narrowing are separate bindings, including
repeated uses of one callback at different source ports. No floating numerical
correspondence has been validated by these frontend observations.

The unchanged schema2 `declaration_scope`, `validate_case` and
`validate_deferrals` checks pass locally. The global exact queue checker derives
its authority and compares the entire queue; appending this JSON alone is
insufficient. `authority_adapter.py` proposes a fail-closed integration hook:
after root review, pin the final package receipt hash and call `extend_authority`
on the existing derived authority. The receipt records the exact active `.lean` to frozen `.lean.txt` mapping; the adapter requires that mapping in checkpoint mode. It verifies the frozen artifacts, exact
current public/helper/source hashes and old/new rule/table uniqueness. Keep
`validate_exact_queue` equality intact; only its reporting counters should become
derived counts. Regenerate the unified queue from the extended authority, then
bind the reviewed catalog and scope ledger. Do not copy the producer's pending
review into a root-reviewed status.

Evidence and reproduction
-------------------------

Run from the repository cwd with the existing toolchain:

```
PATH=/tmp/veritile-lean-toolchain/bin:$PATH LAKE_HOME=/tmp/veritile-lean-toolchain \
  /tmp/veritile-lean-toolchain/bin/lake env python3 -B \
  /tmp/veritile-standardization-20261015/relu-finite-acceptance-wip/verify.py \
  bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteIndices.lean \
  bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteBoundary.lean \
  bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteHost.lean \
  bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteOutcomes.lean \
  bench/tritonbench_g/relu_triton_kernel/FiniteCorrect.lean
```

Every compilation attempt, including failures, retains its exact source snapshot,
command, exit code and hashed log. The initial dependency capture followed the
first draft attempts; it precedes the final validation run. Before/after imported
source/object and toolchain hashes are checked unchanged. The final receipt
freezes all generated evidence and scripts. CPU metadata observations exercise
the exact empty_like call on ten layouts, including unsafe expanded input;
they do not claim CUDA allocation or GPU kernel execution. Repository and Git
were not written by this producer.
