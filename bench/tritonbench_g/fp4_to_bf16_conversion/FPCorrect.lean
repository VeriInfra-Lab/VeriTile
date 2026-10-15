import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPContract
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPSoftware

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Entire repaired candidate for every count, group/block size and program id.
The independent initial-state reference includes the original floor-masked
scale load's undef when the last scale group is partial. All loads precede
all stores, so logical input/scale/output aliasing is unrestricted. ActiveBytes
is uint8 input typing, not a numerical identity. The pow2 and bf16 multiply
callbacks are explicit; no device refinement is asserted. -/
specification fp4_to_bf16_scaled_candidate_backend_correctness
    (rest : Algebra Word) (P : Primitives) (p : Parameters) (initial : State Word)
    (hbytes : ActiveBytes p initial) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial = some final ∧
      final.mem = (written P p initial).mem ∧
      (∀ j : TileIndex [p.block*2], active p initial j →
        final.mem p.output (outputOffset p initial j) = .mk .bf16 (expectedValue P p initial j)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ j : TileIndex [p.block*2],
        active p initial j → o ≠ outputOffset p initial j) → final.mem r o = initial.mem r o) := by
  refine ⟨(execution rest P p initial).val, (execution rest P p initial).property,
    execution_memory rest P p initial hbytes, ?_, ?_⟩
  · intro j hj
    rw [execution_memory rest P p initial hbytes]
    exact readback P p initial j hj
  · intro r o hmiss
    rw [execution_memory rest P p initial hbytes]
    exact frame P p initial r o hmiss

/-- Closed software-bit instance: correctly rounded exact power-of-two to
fp32, explicit fp32-to-bf16 cast and IEEE bf16 multiplication. The original
Python kernel fails overload resolution; this theorem concerns its one-edit
int32-exponent candidate plus the documented local int16 AST adaptation.
The five native configurations compile offline, but native powif/multiply/
cast payloads and address/memory refinement remain unbound. -/
specification fp4_to_bf16_scaled_candidate_machine_correctness
    (rest : Algebra Word) (p : Parameters) (initial : State Word)
    (hbytes : ActiveBytes p initial) :
    ∃ final, FP.Structural.exec (backend rest software) (source p) initial = some final ∧
      final.mem = (written software p initial).mem ∧
      (∀ j : TileIndex [p.block*2], active p initial j →
        final.mem p.output (outputOffset p initial j) = .mk .bf16 (expectedValue software p initial j)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ j : TileIndex [p.block*2],
        active p initial j → o ≠ outputOffset p initial j) → final.mem r o = initial.mem r o) :=
  fp4_to_bf16_scaled_candidate_backend_correctness rest software p initial hbytes

#axiomsClean fp4_to_bf16_scaled_candidate_backend_correctness
#axiomsClean fp4_to_bf16_scaled_candidate_machine_correctness
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPCorrect
