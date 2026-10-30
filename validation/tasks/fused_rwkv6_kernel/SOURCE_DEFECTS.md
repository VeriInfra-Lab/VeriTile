# Checked scale admission exceeds the actual scalar ABI

The current kernel revision has a concrete source/input-contract defect and
an outstanding full-quotation proof connection. Existing mathematical,
scalar-recurrence, memory and historical stage results are retained. This
diagnosis does not change the canonical Python, its input domain or any Lean
statement, and does not count as completed kernel correctness.

## Independent target and precision

For traversal visit `t=n` (forward) or `t=T-1-n` (reverse), each state entry is

```
H_0[e,f] = optional H0[e,f], otherwise 0
O_t[f] = sum_e (H_n[e,f] + K_t[e]*V_t[f]*U[e]) * (Q_t[e]*scale)
H_(n+1)[e,f] = H_n[e,f]*exp(W_t[e]) + K_t[e]*V_t[f]
```

The launched kernel stores one partial output per key shard and optional
terminal state. The existing host `sum(0)` composes key shards. The Real target
is independent of tiling. Native semantics retain padded reduction leaves,
FP32 seed addition `0+H0`, cast/load/store barriers, actual FMA contraction and
exp's `0x3fb8aa3b` multiply followed by approximate exp2. Q/K/V/W/U, state,
scratch and returned output are FP32. The actual raw Python scale has a JIT
scalar type of its own before source arithmetic converts it; float and integer
bindings must not be silently identified.

## Concrete admitted failures and controls

Use `B=H=T=K=V=1`, contiguous FP32 tensors filled with one, no initial/final
state, forward traversal. The exact checked `forward` method accepts every
case below and passes scale unchanged to the captured launch. Its only scale
guard is exact Python type membership in `(int,float)`.

| Raw scale | Actual JIT result |
| --- | --- |
| `0.5` | fp32 binding and SM80 compilation succeed |
| `2` | i32 binding and SM80 compilation succeed |
| `2**31` | i64 binding succeeds |
| `-(2**63)` | i64 binding succeeds |
| `2**63` | u64 binding succeeds |
| `2**64-1` | u64 binding and SM80 compilation succeed |
| `2**64` | binding raises `OverflowError` |
| `2**100` | binding raises `OverflowError` |
| `-(2**63)-1` | binding raises `OverflowError` |
| `float('inf')` | fp32 binding succeeds |

All three failing live cases have the exact diagnostic:

```
integer to be specialized too large to represent
```

The error occurs inside the real JIT argument binder before kernel compilation
or GPU execution. It is not a numerical accuracy question or a device-resource
failure. The public `Checked.PythonNumber.integer` accepts an arbitrary Lean
`Int`; `numberArgument=true` and `_check_scale` impose no representable integer
range, so the current declared scale domain includes these failures.

The source has a meaningful zero-time branch: T=0 with no final state skips the
launch and preserves `scale=2**64` in saved context without binding it. T=0
with a requested final state still launches and the same scale is rejected by
the binder. A contract candidate must preserve this distinction.

Reproduce from the repository root:

```
python bench/tritonbench_g/fused_rwkv6_kernel/check_scale_binding.py
```

`--output /tmp/rwkv-scale-abi-review` selects a separate evidence directory.
The default report is
`.veritile-cache/kernel-evidence/fused_rwkv6_kernel/scale-binding/report.json`.
It pins the canonical source, checker, installed `runtime/jit.py`, native
specializer binary and twelve IR/PTX artifacts for the three legal controls.
The checker executes the actual checked forward wrapper with CPU-backed
Tensor proxies: CUDA identity/context are adapted, while actual shape, dtype,
contiguity, view flags and pointer metadata are retained. The captured pointer
arguments become their underlying CPU tensors for the real JIT binder and
packer. Scalar identity is checked before binding. No GPU run is claimed.

Canonical Python SHA256:
`dfd768e5f6f3c8b439ef7e76b26a1330e2e81c9071c2d4c091fb8b4122221669`.

## Retained proof scope and separate candidate

The current repaired `Kernels.lean` defines a handwritten `Ops`/`Machine`
execution model. Its complete recurrence and physical result proofs are useful
conditional results, but it is not yet a complete launched `triton` quotation
with a proved refinement into that model. Original benchmark quotation proofs
cover materialized single stages; `Proofs/Reuse.lean` documents their scope.
This missing program connection remains a proof task, not a numerical
assumption. No historical theorem is deleted or weakened here.

A separate candidate can add the raw integer ABI guard
`-2**63 <= scale < 2**64` when a launch occurs. It must retain every other
existing input condition, the nonlaunch branch, float Inf/NaN admission and the
raw scale value/type. Calling `float(scale)` would change source argument
semantics and is not this candidate. It still needs the complete shared DSL
execution proof and independent Real/native/Tensor connections; adding a guard
alone does not complete the kernel revision.

Scalar instruction meaning, compiler correspondence and ordinary Tensor/scalar
API binding are model boundaries, not GPU numerical experiment TODOs. The
known invalid integer ABI cases must be classified or checked explicitly.
This packet did not rerun prior Lean builds, the historical CPU suite, GPU
numerical tests or an independent cold replay.
