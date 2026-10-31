import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicLoadSafety
import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicStoreSafety

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat) (acc : Accumulator α)
    (bytes : ∀ cur, cur<stop → BlockBytes fp8 p s cur)
    (h : Ready p s t start acc) (backed : SourceBounds bounds p s) :
    Safety.rangeSafe (engine a) bounds "step" start stop 1 (KernelSource.loop fp8 p) t := by
  apply Safety.rangeSafe_of_invariant (engine a) bounds "step" stop 1 _
    (fun cur u => ∃ acc, Ready p s u cur acc) _ _ start t ⟨acc,h⟩
  · intro cur active u hu
    obtain ⟨acc,ready⟩ := hu
    exact body_safe a fp8 p bounds s u cur acc ready backed
  · intro cur active u hu v ran
    obtain ⟨acc,ready⟩ := hu
    obtain ⟨v',run,ready'⟩ := body_run_reads a fp8 p s u cur acc (bytes cur active) ready
    rw [ran] at run
    cases run
    exact ⟨_,ready'⟩

theorem loop_code_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (acc : Accumulator α)
    (bytes : ∀ cur, cur<steps p → BlockBytes fp8 p s cur)
    (h : Ready p s t 0 acc) (backed : SourceBounds bounds p s) :
    Safety.traceSafeList (engine a) bounds (loopCode fp8 p) t := by
  have safe := range_safe a fp8 p bounds s t 0 (steps p) acc bytes h backed
  cases fp8 <;>
    simpa [loopCode,KernelSource.loop,kernel,TypedSource.kernel,TypedSource.source,arguments,
      ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,Option.bind,numeric,bop,integralFloorDiv,integralMod,steps,
      Broadcast.leftIndex,Broadcast.rightIndex] using safe

/-- Every source statement is safe along the actual successful execution,
including all K blocks, both rotary branches and the selected typed store. -/
theorem kernel_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (bytes : ∀ cur, cur<steps p → BlockBytes fp8 p s cur)
    (backed : SourceBounds bounds p s) : TraceSafe (engine a) bounds (kernel fp8 p) s := by
  obtain ⟨u,hu,ru⟩ := prefix_run a fp8 p s
  obtain ⟨v,hv,rv⟩ := loop_run_reads a fp8 p s u 0 (steps p) (initial a)
    (fun cur _ active => bytes cur active) ru
  obtain ⟨w,hw,rw⟩ := statistics_run a fp8 p s v _ _ rv
  obtain ⟨x,hx,rx⟩ := rotation_run a fp8 p s w _ rw
  unfold TraceSafe
  rw [source_split]
  simp only [List.append_assoc]
  rw [Safety.traceSafeList_append]
  refine ⟨prefix_safe a fp8 p bounds s,?_⟩
  intro t ht
  rw [hu] at ht
  cases ht
  rw [Safety.traceSafeList_append]
  refine ⟨loop_code_safe a fp8 p bounds s u (initial a) bytes ru backed,?_⟩
  intro t ht
  rw [loopCode_run,hv] at ht
  cases ht
  rw [Safety.traceSafeList_append]
  refine ⟨statistics_safe a fp8 p bounds v,?_⟩
  intro t ht
  rw [hw] at ht
  cases ht
  rw [Safety.traceSafeList_append]
  refine ⟨rotation_safe a fp8 p bounds w,?_⟩
  intro t ht
  rw [hx] at ht
  cases ht
  exact stores_safe a fp8 p bounds s x _ rx backed

#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
