# BMM kernel revision

The kernel objective is the existing complete chunked `A @ B.T` formula in
`Target.lean`: matching rank-three/rank-four FP32 A/B, optional literal signed
INT64 sequence equality, tail padding, and a strict-upper zero triangle when
causal. The authoritative admission checks, all nine configurations, the
accepted repaired Python, and historical causal counterexamples are retained.
No finiteness restriction or ID-content condition is added. Empty outputs have
no launch; K=0 has no reduction-body iteration.

`Kernels.lean` contains one readable FP32 main-source template and the final
32-by-64 cleanup DSL. `fp32Source_eq` identifies that main program with the
accepted typed-source program used by the existing proof library. The cleanup
DSL makes the source store's implicit FP32 conversion explicit. Mathematical
Nat arithmetic is only its logical projection; the existing admission-derived
native certificate covers actual signed-int64 controls, masks, and active byte
addresses.

The candidate changes only `pid_ch - pid_c * ngroups` to
`pid_ch - ngroups * pid_c`. Every floating statement, branch, mask, store, launch
configuration and wrapper check is identical. `KernelRewrite.execution` proves
complete typed execution-state equality for arbitrary floating operation
models with no atomic assumptions. `KernelCandidate.native_head` proves the
same multiplication commutation for actual modular 64-bit values.

`KernelExecution` executes both the main kernel and the cleanup DSL at their
actual grid coordinates. It constructs successful source and candidate runs
and proves equality of the entire final typed state. Its cleanup bridge proves
complete memory equality to the existing zero-scatter model, so it preserves
the historical Real/byte-frame proofs without substituting a cleanup event list
for unproved DSL execution.

The native reference in `NativeReference.lean` starts from independent input
arrays, follows the exact K-block order, applies the actual masked loads and
FP32 casts, threads each reached carry through the configured dot recurrence,
and performs signed-ID selection and output conversion. The program/grid
lemmas connect this recurrence to the executed source registers and all stores.
The new public objective is a single grid invocation for any admitted configuration;
no wrapper, allocator, or autotuning implementation is added as an obligation.
`NativeReference.resultGraph` contains no source evaluator or final-state
readback. It is the result of the independent recurrence and destination
scatter, followed by the complete causal cleanup. Its graph interprets each
dot through the existing fixed K8 instruction plan.

`TensorBinding` connects admission metadata to actual `Torch.FP32.Tensor`
objects, their original storage and physical indices, plus ordinary INT64
Tensor API facts for sequence IDs. It proves A/B payload reads, signed sequence
reads and supplied output-byte addresses. It does not implement allocation.
`Correct.bmm_chunk_fwd_real` constructs the complete actual main-grid and
cleanup-grid execution in one Real memory carrier, yielding the independent
full product and the complete output-exterior frame. `KernelReal` reuses the
proved main-program executions, while `CleanupReal` executes the quoted
cleanup DSL in the same Real interpreter and proves its full memory bridge.
There is no allocator, tuning trace, or returned-object obligation in this
public mathematical endpoint. `Correct.bmm_chunk_fwd_tensor_real` binds the
same complete execution to an explicit ideal interpretation of the original
Tensor payloads, physical output word addresses and exterior physical frame.
`FPEquiv.bmm_chunk_fwd_kernel` proves actual source/candidate grid execution,
original input Tensor payload binding, every output byte at the supplied output
Tensor’s physical address, and the complete exterior-byte frame. The old
mathematical host statements are preserved in `Proofs/HostCorrect.lean`;
host/allocator/autotuning statements remain optional historical support;
`native_launches` supplies the source-to-finite-ABI certificate
for every legal tuning and selected configuration.

The literal source is still `acc += tl.dot(a, b)`. A two-argument dot followed by
an outer add is not asserted to be numerically identical to a carried dot.
The selected compiler graph staging is explicit: a pending source-dot node is
consumed by the following source addition, producing the compiler's carried
dot node. The Triton 3.7.1 SM90 receipt records all nine configurations and all
four causal/sequence flag combinations for both source and candidate, plus the
two cleanup compilations. In all 72 main TTIR artifacts, the unique TF32 dot's
C operand is the loop-carried accumulator, and its result is yielded to the
next iteration; no separately rounded `arith.addf` remains. PTX confirms the
fixed MMA/WGMMA K8 families and live carry selection. These are the recorded
runtime-i32 specializations, not exhaustive compiler or hardware verification.

The numerical instruction functions are ordinary local primitive semantics.
The unconditional source/candidate theorem uses the same primitive/compiler
model on both sides, including all floating bit patterns. The older
`bmm_chunk_fwd_if_local` is optional transport between two different primitive
interpretations. It is not a required experiment for the source/candidate
result. Neither the ideal Real theorem nor CPU controls assert TF32 equality
to exact Real arithmetic; padded `0 * infinity`, NaNs, signed zeros and overflow
remain in the native domain.

Validation commands are `check_contract.py`, `check_kernel_controls.py`, and
`check_kernel_lowering.py`. The first two interpret actual Python ASTs with CPU
arithmetic and mocked CUDA launches; each checks 9 admissions and 51 rejections,
complete causal results, both original causal witnesses, identity and byte
frames. The compiler checker also pins the canonical source hash and the exact
single integer-expression candidate edit. Saved compiler artifacts and final
Lean receipts are listed in the frozen workspace delivery packet at
`.veritile-cache/task-deliveries/bmm_chunk_fwd/delivery.json`. It pins the task
sources, actual public import closure, build drivers and import resolution,
Lean artifacts, CPU controls, and the saved compiler artifacts.

The first frozen packet is retained under `round1/`. The revised packet adds
only the Real kernel/cleanup composition, optional host-entry preservation and
public-entry/audit changes; the CPU and compiler evidence is reused unchanged.
