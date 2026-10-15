# vector_addition: 36 mixed fp16/bf16 profiles

This isolated package adds concrete software source execution, per-program
memory readback/frame, ascending complete grid, and defined-domain conversion
policy independence for every ordered integer/bool ↔ fp16/bf16 pair.

`MixedSmallValueCorrect` exposes two genuine `specification` declarations:

- `vector_addition_mixed_small_total_correctness`: the explicit arbitrary
  undefined-integer-conversion policy remains part of the result.
- `vector_addition_mixed_small_defined_correctness`: only active integer-output
  addresses must be in the finite toward-zero range domain. On that domain,
  stored memory is independent of every arbitrary conversion policy. Floating
  output needs no integer-conversion domain premise.

The fp16 route rounds the exact integer directly with Binary16.round, adds
with Binary16.add, and widens to fp32 before integer stores. The bf16 route
widens the bf16 input to fp32, rounds integers directly to fp32, adds there,
and narrows only for a bf16 output. The Word carrier stores half/bfloat payloads
in its low16 bits; its carrier tag is not a numerical fp32 rounding stage.

All36 copied TTIR graphs are checked through loads (no `other` operand),
conversion signedness/width, ordered addition, result-store lineage, original
pointer roots, signed i32 pid.x*1024+arange offsets and signed i32 masks.
These are observations from the prior frozen frontend audit, not a formal
compiler theorem. No new frontend, native backend or GPU execution is claimed.

The total frozen logical coverage is160/217 promoted canonical pairs. The
remaining57 are18 mixed fp64,11 supported fp64 floating pairs and28 semantically
promoted fnuz frontend rejections. Whole-original-program completion remains0;
native storage/address/error/poison/scheduling/host obligations stay explicit
in scope-review.json, separately from numerical correspondence assumptions.

Final verification compiles all six source files in order, checks exact source
and imported project source/olean hashes, audits axioms/noncircularity, and runs
an actual git diff --cached --check on unchanged source bytes in a temporary
index before freezing. The stack-overflow attempt is preserved under
`evidence/failed-attempts/`; its exact log and failed source hashes match the
captured failed run. Local irreducible attributes in the structural proof
prevent unfolding concrete rational algorithms without changing definitions
or theorem statements. The final successful run is recorded separately.
