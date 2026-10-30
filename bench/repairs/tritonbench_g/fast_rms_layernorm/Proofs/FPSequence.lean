import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPGridBackward
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPGridWeight

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPSequence
open VeriTile.Triton FP.Structural
open FPGridForward (Address Heap)
noncomputable section
variable {α : Type} [Inhabited α]

def run (M : Algebra α) (h : Call) (heap : Heap α) : Option (Heap α) :=
  (FPGridWeight.run M h heap).bind (FPGridBackward.run M h)

/-- The full weight grid completes before the input-gradient grid modifies its
private upstream buffer. dW reads the original upstream snapshot. -/
theorem complete (M : Algebra α) (h : Call) (heap : Heap α) :
    ∃ final,run M h heap=some final ∧
      (∀ row : Fin h.rows,∀ j,final (FPGridBackward.address h row j)=FPGridBackward.expected M h heap row j) ∧
      (∀ pid : Fin (Geometry.featureCount h),∀ j,final (FPGridWeight.address h pid j)=FPGridWeight.expected M h heap pid j) ∧
      (∀ addr,(∀ row : Fin h.rows,¬FPGridBackward.writes h row addr) →
        (∀ pid : Fin (Geometry.featureCount h),¬FPGridWeight.writes h pid addr) → final addr=heap addr) := by
  obtain ⟨weights,hw,wo,wf⟩ := FPGridWeight.complete M h heap
  obtain ⟨final,hx,xo,xf⟩ := FPGridBackward.complete M h weights
  refine ⟨final,by simp [run,hw,hx],?_,?_,?_⟩
  · intro row j
    rw [xo]
    apply FPGridBackward.expected_stable M h row weights heap
    intro addr read
    apply wf
    intro pid
    rintro ⟨lane,eq⟩
    subst addr
    simp only [FPGridBackward.reads,FPGridBackward.writes,FPGridBackward.address,
      FPGridWeight.address] at read
    simp at read
  · intro pid j
    rw [xf]
    · exact wo pid j
    · intro row
      rintro ⟨lane,eq⟩
      have regions := congrArg Prod.fst eq
      exact (by decide : ("dw":RegionName)≠"dx") regions
  · intro addr noX noW
    exact (xf addr noX).trans (wf addr noW)

theorem read_roles_preserved (M : Algebra α) (h : Call) (heap : Heap α) :
    ∃ final,run M h heap=some final ∧
      ∀ r : RegionName,r="x" ∨ r="w" ∨ r="r" → ∀ off,final (r,off)=heap (r,off) := by
  obtain ⟨final,he,_,_,hf⟩ := complete M h heap
  refine ⟨final,he,?_⟩
  intro r hr off
  apply hf
  · intro row
    rintro ⟨j,eq⟩
    have same := congrArg Prod.fst eq
    rcases hr with rfl|rfl|rfl <;> simp [FPGridBackward.address] at same
  · intro pid
    rintro ⟨j,eq⟩
    have same := congrArg Prod.fst eq
    rcases hr with rfl|rfl|rfl <;> simp [FPGridWeight.address] at same

#axiomsClean complete
#axiomsClean read_roles_preserved
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPSequence
