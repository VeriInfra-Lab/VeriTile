import bench.tritonbench_g.chunk_delta_fwd.Sources.Program
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem singleton (M : Algebra α) (bounds : RegionName → Nat)
    (stmt : ComputeStmt) (s : State α) :
    traceSafeList M bounds [stmt] s ↔ stepSafe M bounds stmt s := by
  rw [traceSafeList_unfold]
  cases h : step M stmt s <;> simp only [h,traceSafeList_nil,and_true]

/-- Active logical coordinates of the seven actual source descriptors. -/
structure Bounds (p : Parameters) (pid : PID) (bounds : RegionName → Nat) : Prop where
  key : ∀ t k, t < p.T → k < p.K →
    pid.2*(p.T*p.K)+k+t*p.K < bounds "key"
  weight : ∀ t k, t < p.T → k < p.K →
    weightBase p pid+t*p.wTime+k*p.wDim < bounds "weight"
  value : ∀ t v, t < p.T → v < p.V →
    pid.2*(p.T*p.V)+t*p.V+v < bounds "value"
  seed : p.useSeed = Bool.true → ∀ k v, k < p.K → v < p.V →
    pid.2*p.K*p.V+k*p.V+v < bounds "initial"
  history : ∀ c k v, c < p.NT → k < p.K → v < p.V →
    pid.2*(p.NT*p.K*p.V)+c*p.K*p.V+k*p.V+v < bounds "history"
  corrected : ∀ t v, t < p.T → v < p.V →
    pid.2*(p.T*p.V)+t*p.V+v < bounds "corrected"
  final : p.storeFinal = Bool.true → ∀ k v, k < p.K → v < p.V →
    pid.2*p.K*p.V+k*p.V+v < bounds "final"


private theorem trace_of_steps (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ s : State α, stepSafe M bounds st s) (s : State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply traceSafeList_cons_intro
    · exact safe st (by simp) s
    · intro next _
      exact ih (fun st h => safe st (by simp [h])) next

def entryStart (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (((s.setReg "i_k" .nat [] (fun _ => s.pids 0)).setReg "i_v" .nat [] (fun _ => s.pids 1)).setReg
    "i_bh" .nat [] (fun _ => s.pids 2)).setReg "b_h" .real [p.BK,p.BV]
      (fun _ => M.literal (some .fp32) .real 0)

theorem entry_prefix_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    traceSafeList M bounds ((outer p).before.take 4) s := by
  apply trace_of_steps
  intro st member u
  simp [outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD] at member
  rcases member with rfl|rfl|rfl|rfl
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem entry_prefix_run (M : Algebra α) (p : Parameters) (s : State α) :
    run M ((outer p).before.take 4) s = some (entryStart M p s) := by
  simp [outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    entryStart,ofFloat,toFloat]

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
