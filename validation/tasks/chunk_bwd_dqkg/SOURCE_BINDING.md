# Source and native boundary

The accepted `chunk_simple_gla_bwd_kernel_dqkg` AST, including its decorators,
has SHA-256:

`6f8ef806bb0f23d8659e69b71df5b1cae4639c80f5f5054072b04b8950a8f706`

The digest is computed from `ast.dump(..., include_attributes=False)` and is
checked by `check_contract.py`. The current host-only changes require explicit
contiguity and aligned pointers, and check actual pairwise-disjoint data spans
instead of requiring distinct underlying storage objects. The repaired kernel
body and both autotune configurations are unchanged.

`Kernels.lean` spells out the complete accepted repaired source. The public ABI
instantiates FP32 storage throughout. `Whole.kernel` selects the original
arithmetic order. Source proofs cover the value-loop carries, all block loads
and masks, history chunk selection, causal score matrix, final dQ/dK updates,
per-key-tile gate reduction, final-row correction, all stores, every CTA and
their ownership/frame properties. `sourceLayout_eq` relates the literal
q/v/dh stride queries to the canonical layout from admission; `Domain.lean`
derives formed integer/pointer bounds from the same envelope checked in Python.

`DirectNative.initial` reads original physical little-endian FP32 bytes at the
admitted actual bases. Only the preexisting DG fill changes initial output
values. Output serialization uses the same physical allocation table whose
disjointness is derived from the checks and fresh allocator descriptor.
It writes only the three returned spans and proves that every input byte and
every byte outside outputs remains unchanged. Internal `.real` compatibility
tags on gate-stage cells carry `BitVec 32` words and use identity word encoding;
the returned tensor storage is FP32.

The native layer records explicit graphs for scalar operations, exp2 lowering,
ordered reduction layouts/fusion, and ordered m16n8k8 TF32 chains. Each MMA
coordinate has two fixed vectors of eight actual source operands and the
reached accumulator seed. Shuffles describe routing structurally. The only
cross-implementation numerical laws in `InstructionAgreement` are literals,
scalar add/sub/mul/neg/FMA/exp2, and one fixed MMA coordinate. Runtime scale
bits are separate from their Real projection, preserving negative zero.

`DirectNative.CompilerMapping` is an explicit external boundary for the actual
selected compilation. For both four/eight-warp choices it supplies the shape
and ordered layout descriptor selected by the compiler, its dimension/coverage
admission, and exact descriptor binding. The direct endpoint consumes that
mapping for every trial and selected launch. It proves whole source execution
and original-input readback using those profiles; it does not prove the Triton
compiler implementation or GPU binary execution. Packed/shared fragment
routing, source-to-ISA predicate/SSA mapping, synchronization and runtime
realization remain in this compiler boundary. The descriptor has no values,
output state, run-success condition, or whole arithmetic equality field.

No external artifact checker or saved compiler artifacts were run or validated
for this current repair. The inherited finite profile catalog is illustrative,
not an admission restriction and not evidence for unseen specializations.
Production native use needs an actual matching compiler/profile mapping.
The CPU harness checks source semantics and host metadata behavior only.
