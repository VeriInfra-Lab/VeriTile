# Checked int8_matmul_kernel contract

`int8_matmul_kernel_repaired.py` preserves every original helper, autotune
configuration and kernel. The wrapper has an explicit allocated-output
byte-envelope check before launch. A small admission
wrapper requires rank-two strided FP32 A and packed uint8 B on the same current
CUDA device, `A.shape[1] == 4*B.shape[0]`, contiguous A, resolved negative views,
nonnegative strides, naturally four-byte-aligned nonempty FP32 A pointers, and the exact signed32 envelopes from `Whole001.SourceBounds`.
No input copy, kernel option, cast, or launch geometry is changed. B can retain
supported transposed/sliced/broadcast strides and storage offsets. Read-only
input aliases are allowed under the ordinary storage binding.

The conversion guard requires finite A values whose truncations fit `[-128,127]`.
Fractions such as -128.9 and 127.9 are admitted. NaN, infinities, -129 and 128 are
rejected before allocation. M=0/N=0 retain the original empty-grid submission;
K=0 retains zero accumulation. The unchanged per-configuration static assertion
`K % (4*BLOCK_SIZE_K) == 0` remains successful compiler admission. There is no
invented `K % 512` Python requirement: every completed trial and the selected
configuration must satisfy its own assertion.

For row r and column c, the independent mathematical target is

```
sum(f=0..3, q=0..K/4-1,
    truncInt8(A[r, f*(K/4)+q]) * (((B[q,c] >> (2*f)) & 3) - 1))
```

A advances continuously across all four passes; B resets each pass. The
mathematical theorem uses an unbounded integer interpretation. The finite
target instead follows the selected fixed-32-lane instruction recurrence,
including wrapping or saturation as specified by that instruction.

`Proofs/Contract.lean` proves exact guard/domain equivalence and constructs the
accepted whole domain. Exact input byte-envelope checks and the allocated
output check derive `PointerBounds` through `OrdinaryPointers.checked`; ordinary
pointer metadata now supplies only actual allocation-base/data-pointer relations.
The output check can fail after allocation and before any kernel store. `active_a_aligned` derives every active FP32 scalar-load alignment from the checked actual data pointer; packed uint8 B has width one. `Correct.checked_mathematical` and
`FPEquiv.checked_finite` reuse the full `Whole001` proofs for every output cell,
completed trial, returned fresh int32 tensor, and caller frame. Coverage,
physical pointer/byte-span metadata, fresh allocation, successful compilation
and runtime execution retain their ordinary API contracts. No output value is
assumed. `Whole001.whole_native` also retains the accepted local-event ABI
composition. The two numerical leaf families remain explicit and deferred:
defined FP32-to-signed8 conversion and the selected fixed-32-lane live-seed
instruction. No no-overflow premise is added.

The original lazy-negative-view counterexample is preserved. Its pointer-only
call cannot distinguish A=+1 from the contiguous negative view A=-1; B bytes255
produce opposite mathematical signs. This checked revision explicitly rejects
that unresolved view rather than certifying the original wrapper retroactively.

All 17 accepted closure modules and the eight checked/standard-entry modules
compiled with standard-base axiom checks. `python
bench/repairs/tritonbench_g/int8_matmul_kernel/check_contract.py` passes unchanged
kernel/configuration AST checks, 11 admission families, 24 rejection cases, exact allocation,
launch and returned-object checks, propagated allocation/launch failures, and
an actual CPU negative-view pointer/value check. CUDA is mocked; no GPU
numerical validation or historical acceptance change is claimed.
