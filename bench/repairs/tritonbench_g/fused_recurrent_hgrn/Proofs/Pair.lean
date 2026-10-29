import bench.repairs.tritonbench_g.fused_recurrent_hgrn.PairIO
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.IOImplements
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Pair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem suffix_eq (T D BD : Nat) (U S : Bool) :
    (forwardStrideFolded T D BD U S).surfaceBody.drop 6 =
      (forward T D BD U S).surfaceBody.drop 6 := by rfl

private theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α)
    (T D BD : Nat) (U S : Bool) (s : State α) :
    run M ((forwardStrideFolded T D BD U S).surfaceBody.take 6) s =
      run M ((forward T D BD U S).surfaceBody.take 6) s := by
  have folded : bop (fun a b : Nat => a*b) Broadcast.nil
      (bop (fun a b : Nat => a*b) Broadcast.nil (fun _ : TileIndex [] => s.pids 1) (fun _ => T))
      (fun _ => D) =
      bop (fun a b : Nat => a*b) Broadcast.nil (fun _ : TileIndex [] => s.pids 1) (fun _ => T*D) := by
    funext i
    simp [bop, Nat.mul_assoc]
  simp [forwardStrideFolded, forward, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, numeric, bop, natLt, numericEq,
    Region.cast, Option.bind, Nat.mul_assoc]
  rw [folded]
  rfl

private theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (T D BD : Nat) (U S : Bool) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds ((forwardStrideFolded T D BD U S).surfaceBody.take 6) s ↔
      Safety.traceSafeList M bounds ((forward T D BD U S).surfaceBody.take 6) s := by
  simp [forwardStrideFolded, forward, ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
    Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
    Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
    Safety.addressSafe_unfold, Safety.maskActive_unfold, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, numeric, bop, natLt, numericEq, Region.cast,
    Option.bind, Nat.mul_assoc]

theorem forward_run {α : Type} [Inhabited α] (M : Algebra α)
    (T D BD : Nat) (U S : Bool) (s : State α) :
    run M (forwardStrideFolded T D BD U S).surfaceBody s =
      run M (forward T D BD U S).surfaceBody s := by
  calc
    _ = (run M ((forwardStrideFolded T D BD U S).surfaceBody.take 6) s).bind
        (run M ((forwardStrideFolded T D BD U S).surfaceBody.drop 6)) := by
      rw [← run_append, List.take_append_drop]
    _ = (run M ((forward T D BD U S).surfaceBody.take 6) s).bind
        (run M ((forward T D BD U S).surfaceBody.drop 6)) := by rw [prefix_run,suffix_eq]
    _ = _ := by rw [← run_append, List.take_append_drop]

theorem forward_safe {α : Type} [Inhabited α] (M : Algebra α)
    (T D BD : Nat) (U S : Bool) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (forwardStrideFolded T D BD U S).surfaceBody s ↔
      Safety.traceSafeList M bounds (forward T D BD U S).surfaceBody s := by
  conv_lhs => rw [← List.take_append_drop 6 (forwardStrideFolded T D BD U S).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 6 (forward T D BD U S).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  rw [prefix_safe,prefix_run,suffix_eq]


theorem invocation_run {α : Type} [Inhabited α] (p : Ops α)
    (T D BD : Nat) (U S : Bool) (id : Nat × Nat) (s : State α) :
    (invocation p T D BD U S id).run s =
      (IO.invocation p (forward T D BD U S) id).run s :=
  forward_run (algebra p) T D BD U S _

theorem invocation_safe {α : Type} [Inhabited α] (p : Ops α)
    (T D BD : Nat) (U S : Bool) (id : Nat × Nat) (bounds : RegionName → Nat) (s : State α) :
    (invocation p T D BD U S id).Safe bounds s ↔
      (IO.invocation p (forward T D BD U S) id).Safe bounds s :=
  forward_safe (algebra p) T D BD U S bounds _

theorem mapped_run {α : Type} [Inhabited α] (p : Ops α)
    (T D BD : Nat) (U S : Bool) (ids : List (Nat × Nat)) (s : State α) :
    runSchedule (ids.map (invocation p T D BD U S)) s =
      runSchedule (ids.map (IO.invocation p (forward T D BD U S))) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons id ids ih => simp only [List.map_cons,runSchedule,invocation_run,ih]

theorem mapped_safe {α : Type} [Inhabited α] (p : Ops α)
    (T D BD : Nat) (U S : Bool) (ids : List (Nat × Nat)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (ids.map (invocation p T D BD U S)) s ↔
      ScheduleSafe bounds (ids.map (IO.invocation p (forward T D BD U S))) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons id ids ih => simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run,ih]


theorem schedule_run {α : Type} [Inhabited α] (p : Ops α) (observe : α → α)
    (BH T D BDF BDB : Nat) (U S V : Bool)
    (values : ChannelValues α (IO.inputs BH T D)) (s : State α) :
    runSchedule ((variant p observe BH T D BDF BDB U S V).schedule values) s =
      runSchedule ((IO.recurrentIO p observe BH T D BDF BDB U S V).schedule values) s := by
  change runSchedule (_++_) s=runSchedule (_++_) s
  simp only [runSchedule_append,mapped_run]

theorem schedule_safe {α : Type} [Inhabited α] (p : Ops α) (observe : α → α)
    (BH T D BDF BDB : Nat) (U S V : Bool)
    (values : ChannelValues α (IO.inputs BH T D)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds ((variant p observe BH T D BDF BDB U S V).schedule values) s ↔
      ScheduleSafe bounds ((IO.recurrentIO p observe BH T D BDF BDB U S V).schedule values) s := by
  change ScheduleSafe bounds (_++_) s ↔ ScheduleSafe bounds (_++_) s
  simp only [scheduleSafe_append,mapped_safe,mapped_run]



theorem equivalent {α : Type} [Inhabited α] (p : Ops α) (observe : α → α)
    (BH T D BDF BDB : Nat) (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) :
    (IO.recurrentIO p observe BH T D BDF BDB U S V).Equivalent
      (variant p observe BH T D BDF BDB U S V) := by
  intro values domain bounds bound initial binding
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    IOImplements.implements p observe BH T D BDF BDB hF hB U S V
      values domain bounds bound initial binding
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe p observe BH T D BDF BDB U S V values bounds initial).mpr safe
  · exact (schedule_run p observe BH T D BDF BDB U S V values initial).trans ran
  · intro o j active
    rfl

#axiomsClean forward_run
#axiomsClean forward_safe
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Pair
