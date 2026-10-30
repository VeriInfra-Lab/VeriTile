# Finite acceptance boundary for the original programs

The user standard is all source-declared stages, dtypes and branches. Unverified
numerical assumptions may be postponed only with exact statement, operation,
dtype and configuration bindings. This boundary does not accept any program.

Production scope includes source host wiring and kernel execution, all admitted
cases, actual rejected cases, loops/masks/casts, return metadata, memory effects
and frames. State a source-grounded API/target domain; a missing Python assertion
does not imply support for arbitrary Python objects or unknown dtype families.
An already admitted route cannot be removed by narrowing that domain.

Python/Torch metadata, fresh allocation, scalar/tensor primitive operations and
Triton launch/load/store semantics form an explicit trusted boundary. The task
proof must still connect the actual producer/consumer calls, arguments, selected
configuration and results. An unconstrained final-value or launch-success oracle
cannot replace execution. Source-visible view offsets, widths, integer overflow,
unsafe accesses and ownership need exact treatment. A per-task correctness proof
of the compiler, allocator, MMU, scheduler, cache or sanitizer implementation is
outside this acceptance standard. Native correctness is not thereby established.

Test functions remain inventoried and can be marked test_harness only after the
existing non-JIT/production-call review. Import-time random tests do not require
proofs of the entire RNG or module/runtime implementation. Keep genuine
production assertions and first rejection stages; native diagnostic spelling is
not an extra source branch.

Separate faithful typed interpreters are allowed when exact dispatch and the
external memory/result relation compose them. Existing successful logical
execution is retained without requiring a redundant byte-level implementation.
Undefined numerical conversions and known false conclusions cannot be waived as
numerical assumptions. As-written source behavior must remain distinct from an
intended mathematical specification, including ReLU's pid-zero-only stores.

Acceptance still requires exact schema2 stage/dtype/branch coverage, current
public execution statements, numerical registrations where applicable, and an
independent substantive root review. Build, trust audit and official comparator
checks retain their own results. The checker has not been weakened.

The first calibration applies reviewed finite gap replacements only to ReLU,
vector addition and the two quantization tasks. All other current gap records
remain unchanged and must be read under this boundary and refined against
source evidence before acceptance; no source obligations are deleted in bulk.
