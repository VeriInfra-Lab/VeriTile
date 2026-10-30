import bench.repairs.tritonbench_g.attn_fwd_causal.PairIO
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.AttnFwdCausal.PairIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 24000

/-- Same full repaired launch, TF32/FP16 native graph and physical owner
observer; only the head-index remainder expression changes. -/



theorem invocation_run (p : Parameters) (pid : Program) (s : State Native.Expr) :
    (IO.launch Native.algebra Native.dotModel p pid).run s=(invocation p pid).run s := by
  exact (PairConnection.source_pair
    (FP.ProfiledRewrite.engine (withDot Native.algebra Native.dotModel)) p).run _

theorem invocation_safe (p : Parameters) (pid : Program) (bounds : RegionName → Nat) (s : State Native.Expr) :
    (IO.launch Native.algebra Native.dotModel p pid).Safe bounds s ↔ (invocation p pid).Safe bounds s := by
  exact (PairConnection.source_pair
    (FP.ProfiledRewrite.engine (withDot Native.algebra Native.dotModel)) p).safe bounds _

theorem schedule_run (p : Parameters) (work : List Program) (s : State Native.Expr) :
    runSchedule (work.map (IO.launch Native.algebra Native.dotModel p)) s=
      runSchedule (work.map (invocation p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (invocation p pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (p : Parameters) (work : List Program) (bounds : RegionName → Nat) (s : State Native.Expr) :
    ScheduleSafe bounds (work.map (IO.launch Native.algebra Native.dotModel p)) s ↔
      ScheduleSafe bounds (work.map (invocation p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Full two-stage source, every complete grid order, all tagged FP16 words
and both raw exterior frames. No floating operation or local profile changes. -/
theorem equivalent (instructions : Native.Instructions) (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    (NativeIO.program instructions p work).Equivalent (candidate instructions p work) := by
  intro vals _ bounds hb initial hin
  obtain ⟨safe,final,ran,values,frame⟩ :=
    NativePublicIO.implements instructions p apart work legal vals trivial bounds hb initial hin
  refine ⟨safe,(schedule_safe p work bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run p work initial).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttnFwdCausal.PairIO
