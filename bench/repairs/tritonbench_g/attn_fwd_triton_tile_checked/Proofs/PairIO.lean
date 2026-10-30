import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph ReadLocality
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 24000

theorem invocation_run (p : Config) (pid : (ProgramIndex p)) (s : State Expr) :
    (NativeInterface.invocation p pid).run s=(invocation p pid).run s := by
  exact (PairConnection.source_pair
    (FP.ProfiledRewrite.engine Graph.backend) p (AttnFwdTriton.HostShape.blockD p.headDim)).run _

theorem invocation_safe (p : Config) (pid : (ProgramIndex p)) (bounds : RegionName → Nat) (s : State Expr) :
    (NativeInterface.invocation p pid).Safe bounds s ↔ (invocation p pid).Safe bounds s := by
  exact (PairConnection.source_pair
    (FP.ProfiledRewrite.engine Graph.backend) p (AttnFwdTriton.HostShape.blockD p.headDim)).safe bounds _

theorem schedule_run (p : Config) (work : List (ProgramIndex p)) (s : State Expr) :
    runSchedule (work.map (NativeInterface.invocation p)) s=
      runSchedule (work.map (invocation p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (invocation p pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (p : Config) (work : List (ProgramIndex p)) (bounds : RegionName → Nat) (s : State Expr) :
    ScheduleSafe bounds (work.map (NativeInterface.invocation p)) s ↔
      ScheduleSafe bounds (work.map (invocation p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Complete original grid/domain, tagged BF16 output and both raw frames. -/
theorem equivalent (p : Config) (stage : p.stage=3)
    (nonoverlap : NonOverlapping p) (separate : Separated p) :
    (NativeInterface.programIO p).Equivalent (candidate p) := by
  intro vals _ bounds hb initial hin
  obtain ⟨safe,final,ran,values,frame⟩ :=
    NativeInterface.implements p stage nonoverlap separate vals trivial bounds hb initial hin
  refine ⟨safe,(schedule_safe p (NativeInterface.fullGrid p) bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run p (NativeInterface.fullGrid p) initial).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.PairIO
