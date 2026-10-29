# Fused recurrent delta repair

The candidate preserves the accepted two kernel edits: vector-beta `p_dbeta`
includes its global value-feature offset, and only vector-beta scratch is
zero-initialized. All forward, reverse, recomputation and host-sum operations,
source FP32 storage boundaries, flags and the in-place mutation of `v` remain.
The original Python is unchanged. Importing the candidate runs no fixture.

## Checked wrapper domain

Forward requires positive B/H/T/K/V; dense contiguous ordinary FP32 CUDA
q/k/v, optional scalar-per-token or vector-per-value beta, and optional H0.
Shapes/devices must agree, caller byte windows must be disjoint, and the actual
BH/grid, padded source indexing, block-element and native pointer limits are
checked. Neither K nor V must divide a tile. Canonical contiguous strides are checked explicitly, including singleton axes.
Torch `is_contiguous()` alone does not establish the source stride equations.
All formed signed32 expressions,
including masked lanes and final forward/reverse pointer movements, use the
proved `Index.Safe` span. Actual bases are aligned, have enough lower address
space for reverse retreats, and cannot overflow unsigned64 after displacement.
Fresh scratch/output windows are checked again before each kernel submission.

The public output flag must be a Python bool. Explicit scale must be a positive
Python float with finite FP32 launch conversion; small positive values that
underflow to launch +0 remain admitted. Default scale remains Python
`K ** -0.5`, followed by the actual FP32 JIT conversion. The Real contract uses
the exact Real value of this launch scalar, not exact reciprocal square root.
Omitted beta still invokes the source `ones_like` operation.

Backward validates the actual cotangents and saved tensor metadata before
allocation/launch. It admits one ordinary first-order backward, enforced by
`ctx._used` and the grad-mode check. Dense do and optional dHT must satisfy the
same dtype/device/window rules; dHT requires a returned HT. Saved-tensor version
checking is the ordinary autograd boundary. Repeated or higher-order backward,
expanded/noncontiguous cotangents, and aliasing input windows reject.

## Whole result and independent proofs

For pristine x=v, supplied-or-zero H0, and each token:

```
r_t = x_t - H_t^T k_t
u_t = beta_t * r_t
H_(t+1) = H_t + k_t u_t^T
o_t = scale * H_(t+1)^T q_t
```

The source writes r_t into the original v allocation. It returns o[B,H,T,V]
and optional FP32 HT[B,H,K,V]. With terminal G_T=dHT or zero, the independent
full-sequence VJP includes dQ/dK/dV/dBeta and trainable-H0 gradient. The
`vjp_is_derivative` theorem proves the actual derivative of the independent
sequence loss, including initial-state perturbations and terminal cotangents.
Backward returns seven slots: dq, dk, dv, dbeta, None, optional dh0, None.

`Kernels.lean`, `GeometrySource.lean`, `ABISource.lean`, and `Contract.lean`
contain source/metadata declarations without proof-helper imports. Helpers
consume those declarations unchanged. `Proofs/Contract.lean` derives index,
finite-scale, descriptor, disjoint-window and pointer facts from the executable
guard. `Proofs/Host.lean` binds observations to actual input objects, keeps
allocation identity separate from tensor-object identity, retains the squeeze
view's backing, saves the actual input handles/scale, and returns the actual
sum-result objects through equal-dtype/device `to` operations.

`Correct.lean`’s `checked_returned` proves the independent Real primal/VJP and complete
logical frame. `FPEquiv.lean`’s `checked_fp_returned` proves the exact native-source
FP32 bytes, output pair, gradient slots and storage effects. Forward and
backward are separate guarded entries, each retaining its rejection frame;
the composed public theorem passes the actual forward-mutated snapshot to
backward. Allocation byte frames include padding; `Host.global_frame` retains
every unrelated byte of the global heap outside the actual source-write map.
Source current-memory reloads of v/dk/dv pass through the four-byte codec.

Ordinary runtime boundaries are tensor metadata and byte observations, fresh
allocator/squeeze/sum/to handles, saved-context/version checks, compiler-selected
value-independent reduction/FMA plans, and completed kernel submissions. The
proof does not claim a successful completed result after allocation failure,
aborted device execution, or a later host primitive exception. Guard rejection
is modeled explicitly before source execution; a backward rejection preserves
the already-completed forward mutation.

## Numerical deferrals and validation

Only four fixed-arity native instruction correspondences remain conditional:
FP32 add, subtract, multiply, and FMA against the selected software semantics.
The exact compiled reduction trees and contraction routing remain explicit
structural compiler observations. There is no whole-kernel, whole-dot,
whole-output or FP32-equals-Real assumption. These numerical leaves cannot be
universally established by a finite experiment. No native GPU/compiler
conformance measurement was performed.

All restored task modules and guarded public modules compile with standard-base
axiom/noncircularity audits. CPU replay executes the actual unchanged candidate
kernel ASTs over physical CPU tensor storage. It passes 30 guard rejections,
7 complete forward/backward admissions and 77 successful source-program executions,
plus one expected out-of-bounds execution under the former guard,
plus rejection of a repeated backward for each admitted call. Cases include
V=65 vector-beta scratch, nonmultiple key/value widths, default/scalar/vector
beta, absent/detached/trainable H0, optional HT and nonzero terminal gradients.
An independent dense differentiable recurrence checks outputs, in-place
residuals and all gradients within floating tolerance; all allocation padding
is checked exactly. This validates examples and host/source binding, not the
universal native numerical leaves.

### Singleton-axis admission correction

A concrete pre-correction counterexample has q shape `[2,1,2,3]`, element
strides `(6,100,3,1)`, and a 12-word backing. Torch reports this view contiguous
because the H axis has size one. With ordinary k `[2,1,2,3]` and v `[2,1,2,5]`,
all former guards accepted it. The source's flattened `i_bh=1` then forms
q and k pointers using `i_bh * q.stride(1)`, so their active loads would
access offsets 100, 101 and 102 outside the 12-word views. The first actual
read is the k load, where the CPU replay detects the violation. The new canonical-stride guard rejects it before
any launch. The CPU regression temporarily removes only that guard, confirms
old admission, executes the actual kernel AST until the out-of-bounds read,
then confirms the corrected guard prevents execution. No native OOB run was
performed. This corrects the admitted domain; the kernel arithmetic is unchanged.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `469d63b2788ba099221e8bedfe9fd9ddac83d673c5262b938acbf9852e1ce682`.
The former same-device checks did not select Triton's launch device: the
installed JIT reads the current CUDA device and stream. Both source submissions
now enter `torch.cuda.device(q.device)` and restore the caller device. The
formal host's `device` is this effective launch device. Kernel bodies, guards,
arithmetic and source storage effects are unchanged. CPU replay exercises
forward and backward on logical cuda:1 with an ambient cuda:0 caller, asserts
the device at each launch and verifies restoration. Historical acceptance is
retained as evidence; these changed hashes require acceptance correction.
