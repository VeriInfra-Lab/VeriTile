import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicLoadSafety
import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicStoreSafety
import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOSafetySource

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
    Safety.rangeIntSafe (engine a) bounds "step" start stop 1 (KernelSource.loop fp8 p) t := by
  let invariant (q : Int) (u : State α) : Prop :=
    ∃ n : Nat,q=n ∧ ∃ acc, Ready p s u n acc
  apply Safety.rangeIntSafe_of_invariant (engine a) bounds "step" stop 1 _ invariant _ _ start t
    ⟨start,rfl,acc,h⟩
  · intro q active u hu
    obtain ⟨n,rfl,acc,ready⟩:=hu
    exact body_safe a fp8 p bounds s u n acc ready backed
  · intro q active u hu v ran
    obtain ⟨n,rfl,acc,ready⟩:=hu
    obtain ⟨v',run,ready'⟩:=body_run_reads a fp8 p s u n acc (bytes n (by omega)) ready
    rw [ran] at run
    cases run
    exact ⟨n+1,by simp,_,ready'⟩

theorem loop_code_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (acc : Accumulator α)
    (bytes : ∀ cur, cur<steps p → BlockBytes fp8 p s cur)
    (h : Ready p s t 0 acc) (backed : SourceBounds bounds p s) :
    Safety.traceSafeList (engine a) bounds (loopCode fp8 p) t := by
  have safe := range_safe a fp8 p bounds s t 0 (steps p) acc bytes h backed
  rw [loopCode_header]
  have evaluated:=RmsMatmulRbeScalarDomain.range_header (engine a) (headerArguments p) t
  have pureHeader:=RmsMatmulRbeScalarDomain.IOSafety.range_header_safe (engine a)
    (headerArguments p) t bounds
  have stopEval : evalOp (engine a) none
      (RmsMatmulRbeScalarDomain.countExpr (headerArguments p)) t=some (fun _=>(steps p:Int)) :=
    evaluated.2.1
  simpa [Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    pureHeader.1,pureHeader.2.1,pureHeader.2.2,evaluated.1,stopEval,evaluated.2.2] using safe

/-- Every source statement is safe along the actual successful execution,
including all K blocks, both rotary branches and the selected typed store. -/
theorem kernel_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (bytes : ∀ cur, cur<steps p → BlockBytes fp8 p s cur)
    (backed : SourceBounds bounds p s) : TraceSafe (engine a) bounds (kernel fp8 p) s := by
  obtain ⟨u,hu,ru⟩ := prefix_run a fp8 p s
  obtain ⟨v,hv,rv⟩ := loop_run_reads a fp8 p s u 0 (steps p) (initial a)
    (fun cur _ active => bytes cur active) ru
  change rangeInt (engine a) "step" 0 (steps p) 1 (KernelSource.loop fp8 p) u=some v at hv
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
