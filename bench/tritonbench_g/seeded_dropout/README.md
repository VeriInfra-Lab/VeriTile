# seeded_dropout

Source port of TritonBench-G `_seeded_dropout`, pinned to upstream
commit `603e28a`. The integrated 19-module checkpoint has 82 checked standard-axiom
outputs and empty FP assumptions. Shared-tree validation and independent
comparator replay remain separate from those Lean checks.

`Kernels.lean` retains the complete original kernel and a candidate that only
commutes `pid * BLOCK_SIZE`. Runtime seed and p, omitted load padding,
canonical Philox random generation, strict `random > p`, eager division,
`tl.where`, and the masked store remain. Seed/B materialization and the explicit
destination cast are documented source adapters; they require strict source
correspondence checks rather than an audit exemption.

`Correct.lean` gives original/candidate real correctness through the standard
`ComputeCorrect.RealizesFrame_without_Rounding` surface with exact `MemCell`
outputs, conjoined with successful execution over a finite flat allocation.
No existing KernelIO skin packages these runtime seed and storage profiles.
The output is the initial input divided
by `1-p` when the canonical random value exceeds p, and zero otherwise. It
includes successful execution, real-tagged output cells, and preservation of
all other tagged cells, including pointer cells and unallocated addresses.
Per-program input/output aliases are allowed because loads precede stores.
The generic flatten bridge's `u.undef = 0` premise is explicit; source padding
has not been replaced. Real division is total, including p=1.

`Proofs/SourceDomain.lean` proves launch coverage and exact signed-i32 index
arithmetic on every launched lane, including inactive tails. Legal B is a power
of two through `2^20`, and `N < 2^31`; the original wrapper's B=1024 is included.
The original seed must fit its selected i32/u32/i64/u64 ABI for real execution.

`FPEquiv.lean` states complete typed implementation equivalence with an empty
numerical assumption list and no floating operand-domain restriction. Its
proof uses exact natural-index commutation and a shared complete suffix,
preserving full states, failures, canonical random bits and all floating
operations. This equivalence does not certify a physical GPU backend.

`Proofs/TypedExecution.lean` also checks nonempty complete typed runs from empty
registers at N=B=1, PID=0, for all six storage and four seed profiles. Runtime p
and the original valid seed remain arbitrary. Its concrete `nonempty_pair`
uses the standard default-fp32 structural engine over an explicit real-valued
consistency algebra; it witnesses successful active execution, not IEEE results.

Formal storage choices are fp32, fp16, bf16, fp64, f8e4 and f8e5. Separate offline
Triton 3.7.1/sm90 compilation of the unchanged original JIT at the wrapper's
B=1024 accepted all six storage × four seed ABI configurations, keeping
N:i32/p:fp32/seed dynamic. TTIR, TTGIR, LLVM IR, PTX and cubin hashes are in
`/tmp/veritile-seeded-dropout-native-b1024/compile_receipt.json`.
The four fp32 profiles match the previous B=32 TTIR exactly after normalizing
only debug locations and the block-size constant/range/tensor extent.
Compilation does not establish physical backend correctness. No GPU was run;
the six profiles do not characterize every dtype the Python wrapper may accept.

All axiom/assumption printing lives in `Proofs/Audit.lean`. Build receipts,
source hashes, dependency hashes and actual diagnostics are recorded by the
isolated staging harness. See `source_scope.md` for the precise boundaries.
