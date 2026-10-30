import bench.tritonbench_g.parallel_attention.Combined

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- Execute both complete source helpers in the parent kernel's order.
The barrier has no scalar-memory effect; each helper retains its own local
compiler phase, live accumulators, and full source statement execution. -/
def combinedRun (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 64) (s : State Word) : Option (State Word) :=
  (queryRun a d σ p.tile s).bind (keyValueRun a d σ p.tile)

def backwardGridTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (p : Program d.shape 64) (s : State Word) (R : RegionName) (off : Nat) : Cell Word :=
  if R="dq" then queryGridTarget a d σ p s R off else keyValueGridTarget a d σ p s R off

theorem backward_grid (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (σ : ℝ) (s : State Word) :
    ∃ final,runPrograms (combinedRun actual d σ) (programs d.shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs d.shape 64,∀ R off,backwardWrites d.shape p R off → final.mem R off=backwardGridTarget reference d σ p s R off) ∧
      (∀ R off,(∀ p∈programs d.shape 64,¬backwardWrites d.shape p R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (backward_write_unique d.shape) ?_ ?_ _ _
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨eq,_⟩|⟨eq,_⟩ <;> subst R <;>
      simp [queryRectangle,keyRectangle,valueRectangle,outputRectangle] at member
  · intro p before after same R off
    unfold backwardGridTarget queryGridTarget keyValueGridTarget
    split
    · congr 1; funext i; exact queryTarget_inputs reference d σ p.tile before after same i.1 i.2.1
    · split
      · congr 1; funext i; exact keyTarget_inputs reference d σ p.tile before after same i.1 i.2.1
      · congr 1; funext i; exact valueTarget_inputs reference d σ p.tile before after same i.1 i.2.1
  · intro p before
    apply sequential_step _ _
      ((queryRectangle d.shape p.tile).writes 64 d.shape.BK) (keyValueWrites d.shape p)
      (fun t => queryGridTarget reference d σ p t) (fun t => keyValueGridTarget reference d σ p t)
    · intro R off h; exact h.1
    · intro R off h eq
      subst R
      rcases h with h|h <;> simpa [keyRectangle,queryRectangle,valueRectangle,outputRectangle] using h.1
    · intro R off member h
      have eq := h.1
      change R="dq" at eq
      subst R
      simp at member
    · intro t u same R off
      unfold keyValueGridTarget
      split
      · congr 1; funext i; exact keyTarget_inputs reference d σ p.tile t u same i.1 i.2.1
      · congr 1; funext i; exact valueTarget_inputs reference d σ p.tile t u same i.1 i.2.1
    · intro t
      obtain ⟨graph,final,_,executed,reads,frame⟩ := query_native actual reference laws d σ p.tile t
      refine ⟨final,executed,?_,frame⟩
      intro R off write
      rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩
      erw [queryGridTarget,rectangleTarget_at _ _ _ _ active]
      exact reads r e active
    · intro t
      obtain ⟨graph,final,_,executed,keyReads,valueReads,frame⟩ := keyValue_native actual reference laws d σ p.tile t
      refine ⟨final,executed,?_,frame⟩
      intro R off write
      rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩
      · simp only [keyValueGridTarget,keyRectangle,queryRectangle,ite_true]
        erw [rectangleTarget_at _ _ _ _ active]
        exact keyReads r e active
      · simp only [keyValueGridTarget,valueRectangle,outputRectangle,show ¬(("dv" : RegionName)="dk") by decide,ite_false]
        erw [rectangleTarget_at _ _ _ _ active]
        exact valueReads r v active

#axiomsClean backward_grid
#specNonCircular backwardGridTarget avoiding [combinedRun,queryRun,keyValueRun,FP.Structural.exec,runPrograms]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
