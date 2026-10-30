# Source binding and numerical boundary

`bmm_chunk_fwd_repaired.py` retains the accepted `_bmm_chunk_fwd_kernel_repaired`
AST (SHA256 of attribute-free `ast.dump`:
`a5aae42341d16291a7492ebd7fd616091698f88bac14c1556ca590ef05cd6580`).
The source audit checks the whole function including autotune decorators. No
arithmetic load, reduction, comparison, cast, store, tile selection or tuning
configuration was changed in that function. The new final zero pass is separate.

`Source.lean` defines host shapes, the nine configurations, original conditional
layout branch, precision selection, strides and the exact three-axis grid.
`Kernels.source` selects the complete original typed program with `swapped=false`.
Neither source file imports correctness helpers. `Contract.Metadata.strides`
contains the actual queried batch/row/head/last strides (head is 0 for ungrouped
A/B; sequence uses batch/row). Python and Lean require canonical values whenever
an input can be loaded, and last stride 1 for A/B. Thus retained layout operations
are inactive; empty K=0 input strides are modeled literally without a payload
copy. Optional INT64 sequence loads retain signed decoding and −1/−2 sentinels.

The Real proof reuses the original per-program contraction proof, composes all
flat-memory programs and autotuning trials, then executes the actual upper-cell
zero scatter. Every lower coordinate is active in each original configuration.
`Cleanup.image` gives complete upper coverage, `writer_unique` proves one writer
per cell, and `Native.addresses_exact` connects signed INT64 pid splitting,
aranges, comparisons and pointer arithmetic to that same store list. The repair
therefore establishes the full target, rather than identifying source tile
activity with mathematical causality.

The FP state decodes original A/B bytes little-endian into raw FP32 graph leaves,
and sequence bytes into signed INT64 values. It runs the full typed source
prologue, setup, K loop and epilogue at every actual grid coordinate with mutable
memory threaded through trials. Source numerical operations build expressions;
`NativeProgram.Certificate` proves native scalar controls, per-iteration pointer
state, literal mask equality, active load/store byte bounds and store frames for
all reached states. Certificates are constructed from checked metadata and
ordinary allocation facts. They are not supplied as successful-execution or
pointer-safety assumptions.

Input FP32 loads use the original DSL `.real` compatibility channel. Output
stores and readback use the distinct `.fp32` typed channel. The returned physical
heap is the concrete four-byte serialization of final FP32 words in the fresh
output span and the unchanged original heap elsewhere. The new zero pass uses
raw FP32 `0x00000000`, so upper bytes are proved zero without a numerical port.
Allocator identity/freshness, address-space capacity and separation describe the
ordinary successful allocation boundary. Runtime/compiler failures are not
replaced by a theorem of successful hardware launch.

The public `DirectFP.Arithmetic` numerical ports are literal, cast, binary and
`mma8`. The last has a fixed instruction-family/output-coordinate signature:
two `Fin 8 → BitVec 32` operand vectors, one `BitVec 32` carry and one live bit.
It cannot accept a variable source tile or an entire reduction. `interpret`
structurally expands every consumed dot into the `K8.plan` recurrence;
`body_k8` binds that expansion to the actual loaded tile registers and the
actual old/new accumulator after executing the complete typed source body.
`operands`, `partition`, `unique`, `lane_inside` and `output_coordinate` prove
operand mapping, complete nonduplicated K coverage, absence of synthetic K
padding for every configuration, and output-coordinate reconstruction.

For configurations 0–5 the selected instructions are WGMMA64×N×8; for 6–8
they are MMA16×8×8. The source K32/K64 tile becomes four/eight ordered K8
steps for each output coordinate. The reached carry is live at every step.
This is an explicit lowering selector, not a test of the carry's floating value.
An offline Triton 3.7.1 SM90 compilation of the unchanged FP32 source confirms
all nine instruction families; every WGMMA uses a predicate assigned `-1`,
and every MMA uses its explicit C operand. `compiler_k8_receipt.json` records
source/TTIR/TTGIR/PTX hashes, instruction names and predicate definitions.
The compiler artifacts are saved at the paths in that receipt under `/tmp`.
The recorded specialization has causal and sequence flags false; those flags
control branches/filtering outside the selected dot body. This compiler check
is structural evidence, not a proof of hardware or compiler numerical behavior.

`TaskNumerics` recursively requires only the primitive-local equalities at
actual operands and reached carries. `K8.transport` proves the complete
accumulation recurrence by induction. There is no whole-dot equality, whole-call
success, whole-result equality, ideal Real accuracy law, storage-leaf
correspondence or structural safety assumption in that numerical condition.
Concrete TF32 primitive correspondence remains numerical evidence to supply;
it is not established by the CPU harness or arbitrary-Real theorem.
