# Checked recurrent RWKV6 forward

**Current kernel revision: source/input-contract defect pending repair.** The
exact-type scale guard admits raw Python integers outside the actual JIT ABI;
the checked wrapper accepts them but binding fails before compilation.
[SOURCE_DEFECTS.md](SOURCE_DEFECTS.md) gives actual wrapper/binder negatives,
legal compiled controls, the zero-time branch distinction and retained proof
scope. The canonical Python and all existing Lean files remain unchanged.

This repair covers `fused_recurrent_rwkv6` and the source's direct
`FusedRecurrentRWKV6Function.forward`, including reverse traversal for the latter.
The original has no backward or JVP implementation. The original benchmark file
is unchanged. The kernel changes one expression: the scratch output's head pitch
uses its allocated `T * V`, rather than borrowing the input V head stride.

The supported contract is ordinary strided, contiguous FP32 CUDA tensors on one
device, with resolved negative/conjugate flags. R/K/W have shape `[B,H,T,K]`, V
has `[B,H,T,V]`, U has `[H,K]`, and optional H0 has `[B,H,K,V]`. B/H/K/V are positive;
T may be zero. Read-only inputs may alias; nonzero storage offsets and arbitrary
unused singleton strides are supported. Inputs are passed directly, without
payload copies. Canonical head pitches are supplied explicitly. The two Boolean
options must be exact Python bools; scale must be an exact built-in int or float.
The helper retains its `scale == -1` default selection. Python scalar evaluation
and JIT scalar binding are ordinary local runtime boundaries.

The wrapper checks the actual tile choices, each CUDA grid axis, a conservative
Python-integer bound on all source setup expressions, and alignment/positive and
negative address room for nonempty input and scratch/state pointers. It runs in
the owning CUDA device context. T=0 with no final state skips the launch. T=0 with
a final state executes the original FP32 seed addition: `0 + H0`, including its
signed-zero behavior. The grid-product compatibility helper remains a shared
follow-up; this task adds no launch segmentation.

The independent public proof roots are:

- `Correct.rwkv_real`: full exact-real recurrence/output and optional terminal
  state, actual returned Tensor descriptors, original saved Tensor handles,
  selected scale/reverse, caller device, and the entire logical-memory frame.
- `FPEquiv.rwkv_native`: complete returned FP32 words and all four bytes per word,
  optional final-state words/bytes, actual original and returned Tensor data
  pointers, exact saved context, input bytes and every exterior byte.
- `FPEquiv.rwkv_native_if_local`: the same result under four fixed-arity local
  add/multiply/FMA/approximate-exp2 instruction correspondences.
- `FPEquiv.rwkv_helper_native`: the same full result for the public helper's
  forward traversal and default-scale selection.

`Checked.admit` is a finite metadata/numeric predicate. `geometry_iff`,
`admit_iff`, and `CheckedIndices.setup_bounds` derive the historical execution
domain from those actual checks. The returned result also carries the pointer
room conclusions for input/scratch/state addresses. Original data pointers
include storage offsets. Ordinary Tensor metadata/contiguous-storage facts,
allocator freshness and capacities, and successful runtime launch/binding are
interface boundaries; none contains a computed output equation.

The native bridge evaluates the retained current-memory source graph and emits
its finite physical byte stores. Its independent scalar target is not the
execution graph. The key reduction includes all padded lanes and the host
`sum(0)` includes every key shard; legal reduction trees are fixed before values.
The numerical model retains FP32 load/store barriers, source seed addition,
local FMA contraction and operand order, and exp's multiply by `0x3fb8aa3b`
followed by `ex2.approx.f32`. Every input FP32 bit pattern is admitted. No equality
between finite precision and exact real arithmetic is assumed. Compiler/source
lowering and scalar ISA meaning remain explicit model boundaries; this is
not a proof of Torch or the compiler implementation.
These boundaries are not numerical experiment TODOs. The existing handwritten
register-loop model still needs a full shared DSL quotation/refinement for the
new kernel-only standard; the historical stage results remain retained.

The historical source-stage proofs are retained in `Proofs/Reuse.lean`, and the
original gapped-head output-address counterexample is retained in
`Proofs/Source.lean`. Generic source, index, recurrence, ownership, full grid,
host sum, native scalar graph and finite-byte trace proofs are reused. The Real
public closure does not import the native endpoint, and the native public
closure does not import the Real correctness or reference modules.

Validation (CUDA unavailable):

- `python check_scale_binding.py`: current-source exact wrapper/JIT scalar
  admission controls, including three live invalid integers, the T=0 launch
  distinction, and three legal compiled scalar-type controls. See the defect
  note and frozen report for scope and hashes.
- The following checks are retained historical evidence, not newly replayed
  or promoted into kernel-revision completion by this diagnosis.

- All 21 Lean modules compile; public axiom and independent-target audits pass.
- `python bench/repairs/tritonbench_g/fused_rwkv6_kernel/check_contract.py`:
  44 host admissions, 27 rejections, helper sentinel, and nine complete CPU
  source-interpreter cases. These include forward/reverse, split K/V, optional
  states, T=0, signed zero, original-handle preservation, aliases, singleton
  strides, nonzero offsets and input bit frames. The original output overflow
  control is reproduced arithmetically without executing an invalid store.
- `python bench/repairs/tritonbench_g/fused_rwkv6_kernel/check_cuda_lowering.py`:
  eight actual-JIT-binder SM80/SM90 profiles, including 1/8/17/32/33/65 feature
  dimensions and empty time, compile offline. Generated artifacts live under
  `/tmp/veritile-rwkv-lowering`.
- `python bench/repairs/tritonbench_g/fused_rwkv6_kernel/check_precision.py`:
  pinned source/original hashes and exact kernel AST delta, no new payload
  copies, independent import closures, actual-JIT artifacts, live FP32
  add/FMA/exp2 dataflow, and no FTZ or MMA in those profiles.

The CPU interpreter tests mathematical behavior and memory framing. Offline
compilation tests the selected backend profiles. Neither establishes GPU
execution or universal scalar ISA/compiler conformance.
