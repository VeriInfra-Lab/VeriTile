import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceAccessBody
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceBounds

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Reuse the existing loop invariant and actual body execution, checking
accesses at every concrete successor state of the signed source loop. -/
theorem loop_safe {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (c : SourceBody.Coordinates) (s t : State α) (start : Nat)
    (acc : Values α .real [32,32]) (h : SourceBody.Ready p c s t acc)
    (bounds : RegionName → Nat)
    (hb : ∀ block, block < (p.inner+15)/16 → BodyBounds bounds p c block) :
    Safety.rangeIntSafe (SourceBody.install M dot) bounds "k" (start : Int)
      (((p.inner+15)/16 : Nat) : Int) 1 (SourceParts.body abi p) t := by
  rw [Safety.rangeIntSafe_unfold]
  simp only [one_ne_zero,↓reduceIte,Nat.cast_lt]
  by_cases hs : start < (p.inner+15)/16
  · simp only [hs,↓reduceIte]
    refine ⟨body_safe M dot abi p c s t start acc h bounds (hb start hs),?_⟩
    obtain ⟨u,runBody,ready⟩ := SourceBody.body_run M dot abi p c s t start acc h
    rw [runBody]
    simpa only [Nat.cast_add,Nat.cast_one] using
      loop_safe M dot abi p c s u (start+1) (SourceBody.update M dot abi p c s start acc)
        ready bounds hb
  · simp only [hs,↓reduceIte]
termination_by (p.inner+15)/16-start
decreasing_by omega

theorem loop_start (abi : SourceABI) (p : KernelArgs) :
    (SourceParts.loopParts abi p).start = .constInt 0 := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem loop_stride (abi : SourceABI) (p : KernelArgs) :
    (SourceParts.loopParts abi p).stride = .constNat 1 := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem loop_step_safe {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (c : SourceBody.Coordinates) (s t : State α)
    (acc : Values α .real [32,32]) (h : SourceBody.Ready p c s t acc)
    (bounds : RegionName → Nat)
    (hb : ∀ block, block < (p.inner+15)/16 → BodyBounds bounds p c block) :
    Safety.stepSafe (SourceBody.install M dot) bounds
      (.forRangeInt "k" (SourceParts.loopParts abi p).start (SourceParts.loopParts abi p).stop
        (SourceParts.loopParts abi p).stride (SourceParts.body abi p)) t := by
  rw [Safety.stepSafe_unfold]
  refine ⟨?_,?_,?_,?_⟩
  · rw [loop_start,Safety.safeOp_unfold]; trivial
  · rw [SourceBounds.stop_common]
    simp [SourceParts.loopParts,SourceParts.active,SourceParts.afterTile,SourceParts.afterLength,
      SourceParts.thenBody,SourceParts.outer,Source.intRange?,kernel,primaryHalf,sourceFace,
      sourceHalfSingle323232,ComputeKernel.surfaceBody,Safety.safeOp_unfold]
  · rw [loop_stride,Safety.safeOp_unfold]; trivial
  · rw [SourceBounds.eval_start,SourceBounds.eval_stop _ _ _ _ h.inner,SourceBounds.eval_stride]
    exact loop_safe M dot abi p c s t 0 acc h bounds hb

#axiomsClean loop_safe
#axiomsClean loop_step_safe
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
