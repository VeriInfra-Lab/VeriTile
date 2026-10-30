import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelExecution
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem trace_of_steps (a : Operations α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st ∈ code, ∀ t, Safety.stepSafe (engine a) bounds st t)
    (s : State α) : Safety.traceSafeList (engine a) bounds code s := by
  induction code generalizing s with
  | nil => simp
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro
    · exact safe st (by simp) s
    · intro t ht
      exact ih (fun st h u => safe st (by simp [h]) u) t

/-- Only the actual masked source reads and stores require backing. -/
structure SourceBounds (bounds : RegionName → Nat) (p : Parameters) (s : State α) : Prop where
  x : ∀ cur (i : TileIndex [16,64]), i.2.1.val<p.K-cur*64 → xOffset p s cur i<bounds "X"
  weight : ∀ cur (i : TileIndex [64,64]), i.1.val<p.K-cur*64 → wOffset p s cur i<bounds "W"
  rms : ∀ cur (i : TileIndex [1,64]), i.2.1.val<p.K-cur*64 → rOffset cur i<bounds "RMS"
  output : ∀ i, active p s i=Bool.true → outputOffset p s i<bounds "OUT"

theorem prefix_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine a) bounds (prelude fp8 p) s := by
  apply trace_of_steps
  intro st member t
  cases fp8 <;>
    simp only [prelude,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
      ComputeKernel.surfaceBody,List.take,List.drop,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem statistics_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine a) bounds (statistics fp8 p) s := by
  apply trace_of_steps
  intro st member t
  cases fp8 <;>
    simp only [statistics,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
      ComputeKernel.surfaceBody,List.take,List.drop,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem rotation_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine a) bounds (rotation fp8 p) s := by
  apply trace_of_steps
  intro st member t
  cases fp8 <;>
    simp only [rotation,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
      ComputeKernel.surfaceBody,List.take,List.drop,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,true_and,evalOp_unfold]
  all_goals split
  all_goals try trivial
  all_goals apply trace_of_steps
  all_goals intro st member u
  all_goals simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

#axiomsClean prefix_safe
#axiomsClean statistics_safe
#axiomsClean rotation_safe
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
