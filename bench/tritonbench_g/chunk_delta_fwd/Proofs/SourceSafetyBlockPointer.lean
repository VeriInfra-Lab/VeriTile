import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
open VeriTile Triton FP.Structural Safety
noncomputable section

/-- Discharge the finite offset list before recursively checking its concrete
expressions. This is the shared safety definition, with no runtime premise. -/
@[simp 10000] theorem safeBlock2 {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (bounds : RegionName → Nat) (s : State α)
    (r : RegionName) (base row col : Op .nat []) (parent block strides : List Nat) :
    safeOp M precision bounds s (.makeBlockPtrDynOffsets r base parent block strides [row,col]) ↔
      safeOp M precision bounds s base ∧ safeOp M precision bounds s row ∧ safeOp M precision bounds s col := by
  rw [safeOp_unfold]
  constructor
  · rintro ⟨hb,offsets⟩
    exact ⟨hb,offsets row (by simp),offsets col (by simp)⟩
  · rintro ⟨hb,hr,hc⟩
    refine ⟨hb,?_⟩
    intro offset member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · exact hr
    · exact hc

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
