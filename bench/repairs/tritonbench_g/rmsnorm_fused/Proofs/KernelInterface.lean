import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Runtime

namespace VeriTile.Bench.Repairs.RmsnormFused.KernelInterface
open VeriTile Triton FP.Structural
noncomputable section

/-- Complete source execution, initial-input values and the entire exterior
frame over raw kernel geometry, scalar operands and typed memory. -/
theorem complete {α : Type} [Inhabited α] (A : Algebra α) (shape : Shape)
    (valid : shape.Valid) (epsilon : α) (schedule : List Nat)
    (all : schedule.Perm shape.grid) (initial : State α) :
    ∃ final, Runtime.run A "X" "Y" "W" shape.rows shape.width shape.block
      epsilon schedule initial=some final ∧
      (∀ row col,row<shape.rows → col<shape.width →
        Runtime.read final "Y" (row*shape.width+col)=
          Runtime.value A initial "X" "W" row shape.width shape.block epsilon col) ∧
      (∀ (region : RegionName) offset,region≠"Y" ∨ shape.rows*shape.width≤offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,hm⟩ := Runtime.source_complete A "X" "Y" "W"
    shape.rows shape.width shape.block epsilon valid.1 valid.2.1 schedule initial
  refine ⟨final,ran,?_,?_⟩
  · intro row col hr hc
    unfold Runtime.read
    rw [hm]
    exact Runtime.whole_value A shape valid "X" "Y" "W" epsilon schedule all
      initial row col hr hc (by decide) (by decide)
  · intro region offset outside
    rw [hm]
    apply Runtime.grid_frame
    rcases outside with different|outside
    · exact .inl different
    · right
      intro row hr col hc equal
      have rowBound := (List.mem_range.mp (all.mem_iff.mp hr))
      change row<shape.rows at rowBound
      have bound : row*shape.width+col<shape.rows*shape.width := by nlinarith
      omega

#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsnormFused.KernelInterface
