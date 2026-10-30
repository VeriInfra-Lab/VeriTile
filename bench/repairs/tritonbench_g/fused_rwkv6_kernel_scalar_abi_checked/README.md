# RWKV6 scalar ABI checked candidate

The selected Python source is `fused_rwkv6_kernel_scalar_abi_checked.py` and the
existing comparison candidate is `fused_rwkv6_kernel_u_early.py`. Both expose
`fused_recurrent_rwkv6_fwd_kernel`. The only candidate body change moves the
three U pointer/mask/load statements immediately after the head ID. No measured
speedup or numerical rewrite is claimed.

`Kernels.lean` and `CandidateKernels.lean` paste those complete function bodies
into ordinary `triton { ... }` blocks. Each block has all 16 top-level Python
statements, including the optional state branches, every cast, signed reverse
pointer expression, time loop and stores. Caller dtypes, named regions and
constexpr arguments are outside the pasted body. `jit-correspondence.json`
checks ten mappings: original/candidate for FP32, I32, I64, U64 and integer 1.
Integer 1 is Triton's native constexpr specialization; float 1.0 stays FP32.

## Input contract and mathematical target

The existing wrapper checks are unchanged. On the actual launch branch
`T != 0 or output_final_state`, raw integer scale must satisfy
`-(2**63) <= scale < 2**64`. Floats, including Inf and NaN, retain their existing
admission. With T=0 and final state disabled, no JIT binding occurs and arbitrary
integers remain admitted. `Contract.lean` records this same branch-local domain;
no conversion to Python `float` or new pitch restriction is introduced.

For traversal step n, let t=n in forward mode and t=T-1-n in reverse mode.
The independent Real recurrence is:

```
O[ik,z,t,f] = sum over this key shard e:
  (H[n,z,e,f] + K[z,t,e]*V[z,t,f]*U[z%H,e]) * (Q[z,t,e]*scale)
H[n+1,z,e,f] = H[n,z,e,f]*exp(W[z,t,e]) + K[z,t,e]*V[z,t,f]
```

The source seed preserves `0 + load(H0).to(float32)` when enabled. Otherwise
it is zero. The optional final store writes H[T]. Native semantics retain
source casts, the nested FP32 query-storage load, padded reduction lanes,
operand order and local compiler choices. The Real formula does not assert
that native floating arithmetic equals real arithmetic.

## Source-to-proof connections

- `IO.lean` owns the complete logical channels, affine head pitches, masks,
  scalar binding and launch schedule. `Runtime` selects actual source/candidate
  bodies for each scalar ABI and reuses this single IO layout. Integer bodies
  retain conversion at their source arithmetic expression.
- `Proofs/Execution.lean` through `Output.lean` connect entry, signed time loop,
  per-step output and optional final store to actual execution. The Nat pointer
  channel is proved equal to signed source addresses at every live access.
  Dead reverse/T=0 and final-backedge pointer values are handled by termination
  and exit noninterference; they are not assumed to be live valid addresses.
- `Proofs/ScalarSource.lean`, `ScalarTransport.lean` and `ScalarSafety.lean`
  connect each full raw integer body, including conversion, execution failure
  and access safety, to the common proof interface. `NativeScalar.lean` binds
  conversion to the explicit scalar ABI primitive, without a cast identity.
- `Proofs/OptionalState.lean` and `RuntimeScalar.lean` cover absent H0/HT using
  zero accessible capacity when their source flags are disabled. No accessible
  allocation is fabricated for a Python `None` argument.
- `Correct.lean` exposes the independent Real formula. `rwkv_real` retains its
  existing signature and FP32 source interpretation; `rwkv_real_abi` covers all
  actual scalar source profiles using the same complete logical IO judgment.
- `FPEquiv.lean` compares `Runtime.original` and `Runtime.earlyU` with `≡[[]]`.
  Both use the raw scalar-selected source body, original complete grid, all
  payloads and optional branches. `PairPrefix` and `ScalarPair` prove the actual
  reordered prefix and unchanged suffix preserve execution and trace safety.
  `RuntimeScalar.equivalent` includes safe successful runs, typed output cells
  and the exact exterior frame.

The original proof scopes remain limited to this forward kernel's key-shard
partials and optional terminal state. Torch `o.sum(0)`, allocation execution,
autograd and runtime scheduling implementation are outside them. The original
repair's diagnosis, host results and optional physical/byte/Tensor endpoints
remain in their existing directories; they are not additional numerical claims.

## Validation boundaries

The raw-body checker and its mutation tests cover all ten source mappings,
including cast and optional-branch deletion, reverse backedges, and exact
integer-1 specialization. Shared frontend execution regressions exercise
augmented conditional syntax, nested aliased loads, signed pointer movement
and preservation of runtime integer conversions versus constexpr literals.
Lean build, theorem-footprint and independent rebuild receipts are recorded in
the repository review checkpoint when this migration is complete; source
correspondence by itself does not certify the proofs.

There are no registered numerical experiment assumptions in the structural
comparison. Scalar ABI conversion, primitive instructions, compiler choices
and reduction plans remain explicit software-model boundaries. Retained
specializer/compiler receipts are historical evidence, not a new GPU run.
The official comparator and Torch-dependent checks must be reported separately
when unavailable; neither is replaced by an invented PASS. No new GPU or
numerical experiment is part of this source-body migration.
