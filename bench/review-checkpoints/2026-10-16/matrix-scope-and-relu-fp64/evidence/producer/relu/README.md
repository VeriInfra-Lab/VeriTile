# ReLU fp64 source route and finite host dispatch

This isolated package adds four Lean modules; it changes no repository source
and does not rewrite the twelve existing typed routes. All files and compilation
attempts are retained under this directory. Original-program acceptance remains
pending and the acceptance increment is zero.

`FP64Source` interprets the seven original statements in source order. The load,
ordered comparison and select retain `BitVec 64`. The input pointer view is f64;
the output pointer view is f32. The comparison implements ordered >= +0 by
binary64 sign/exponent/fraction, accepting both zero signs and rejecting NaNs.
The only numerical callback is the total f64-to-f32 store narrowing operation.
Native numerical correspondence is unverified. The proposed exact bindings in
`evidence/numerical-bindings-proposal.json` are not integrated queue entries.

All pids execute address, mask, load and f64 result statements before the gate.
Inactive lanes use arbitrary supplied f64 words; no implicit zero padding is
introduced. The store reads actual offset, mask and result registers, narrows to
f32, and writes only for pid zero. The readback proof establishes injectivity
of the source's offsets. The independent memory formula depends only on initial
memory. Typed memory uses total f64/f32 views of disjoint input/output regions;
the applicability of those views is an explicit host primitive boundary.

`FP64Grid` passes each entire returned state to the next program. It proves the
memory formula for arbitrary pid lists and therefore every permutation of the
complete ceil-grid. Before/after traces retain every pid and prove its source
execution and f64 registers. The fixed-1024 registers are tied to the existing
signed-i32 arithmetic lemmas for `0 <= N < 2^31`, including inactive lanes.
Every fp32 output cell outside `min N 1024` and every f64 input cell is unchanged.
For N=0 the complete grid is empty. The known original source bug is preserved:
when N>1024 an initially incorrect tail remains incorrect.

`FP64Host` dispatches the frozen seventeen canonical dtypes. The old twelve
successful branches call the existing `TypedValue` source/grid theorem; fp64
calls the new source interpreter. Two fn/fp8 formats are rejected at ordered
comparison and two fnuz formats at signature admission, using the inherited
frontend evidence and a named trusted frontend primitive classification. Those
branches return an error at launch and cannot append the returned-output event.
No additional native compilation or GPU execution was performed. The old audit
was N=1025, block=1024, Triton 3.7.1, Torch 2.13.0+cu130, CUDA sm90. Its observations
are not new native execution evidence for arbitrary N or other targets.

The host runtime is a product of the old `FP.Structural.State alpha` and the new
typed binary64 memory. A dtype branch selects one carrier; the external theorem
projects the old branch's `final.1.mem` or fp64 branch's `final.2` and proves the
unselected carrier unchanged. This is finite dispatch composition, with no
assertion that the two carriers are interchangeable encodings of one memory.
Both successful host proofs derive launch success and invoke the existing
`HostWrapper.relu_success`. Earlier device/allocation/numel outcomes remain
explicit primitive hypotheses, not assumed kernel results.

Remaining finite work for original-program acceptance: bind these outcomes to
the actual Torch tensor metadata/freshness and valid typed input/output views;
bind a justified supported configuration and dtype domain and frontend rejection
classification; integrate and review exact numerical registrations and schema2
public scope evidence. Existing helper/module-stage evidence must be bound as
appropriate. The existing nine integer/bool byte proofs and twelve typed routes
remain available. No new generic compiler/runtime/harness verification obligation
is introduced. This package does not establish whole-tensor mathematical ReLU.

Public declarations:

- `FP64Correct.relu_fp64_source_grid_correctness`
- `FP64Correct.relu_fp64_host_source_correctness`

Verification uses the existing Lean toolchain and imported project objects.
Every real attempt has a source snapshot, command, exit code and hashed log.
Final reproduction (from the repository working directory):

```sh
python3 -B /tmp/veritile-standardization-20261015/relu-fp64-source-wip/freeze.py before
PATH=/tmp/veritile-lean-toolchain/bin:$PATH LAKE_HOME=/tmp/veritile-lean-toolchain /tmp/veritile-lean-toolchain/bin/lake env python3 -B /tmp/veritile-standardization-20261015/relu-fp64-source-wip/verify.py bench/tritonbench_g/relu_triton_kernel/Proofs/FP64Source.lean bench/tritonbench_g/relu_triton_kernel/Proofs/FP64Grid.lean bench/tritonbench_g/relu_triton_kernel/Proofs/FP64Host.lean bench/tritonbench_g/relu_triton_kernel/FP64Correct.lean
python3 -B /tmp/veritile-standardization-20261015/relu-fp64-source-wip/freeze.py after
python3 -B /tmp/veritile-standardization-20261015/relu-fp64-source-wip/freeze.py finalize
```

`finalize` refuses to overwrite a receipt. Reproduction after freezing should
use a separate copy and preserve the original frozen evidence.
