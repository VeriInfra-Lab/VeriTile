import bench.tritonbench_g.fp4_to_bf16.Proofs.FPContract

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The entire unscaled e2m1 kernel, for arbitrary count, block and program id.
Each active uint8 input contributes its high nibble then its low nibble, as
independent initial-memory bf16 table values. Input/output aliasing is allowed
within this program: all input loads precede stores. ActiveBytes is storage
typing, not an extra numerical law; masked undef values remain unrestricted.
This is a closed software-bit contract; device refinement is not asserted. -/
specification fp4_to_bf16_unscaled_machine_correctness
    (rest : Algebra Word) (p : Parameters) (initial : State Word)
    (hbytes : ActiveBytes p initial) :
    ∃ final, FP.Structural.exec (backend rest) (source p) initial = some final ∧
      final.mem = (written p initial).mem ∧
      (∀ j : TileIndex [p.block*2], active p initial j →
        final.mem p.output (outputOffset p initial j) = .mk .bf16 (expectedValue p initial j)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ j : TileIndex [p.block*2],
        active p initial j → o ≠ outputOffset p initial j) → final.mem r o = initial.mem r o) := by
  refine ⟨(execution rest p initial).val, (execution rest p initial).property,
    execution_memory rest p initial hbytes, ?_, ?_⟩
  · intro j hj
    rw [execution_memory rest p initial hbytes]
    exact readback p initial j hj
  · intro r o hmiss
    rw [execution_memory rest p initial hbytes]
    exact frame p initial r o hmiss

#axiomsClean fp4_to_bf16_unscaled_machine_correctness
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPCorrect
