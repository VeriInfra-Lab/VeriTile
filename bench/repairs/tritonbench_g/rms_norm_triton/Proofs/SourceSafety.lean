import bench.repairs.tritonbench_g.rms_norm_triton.IO

namespace VeriTile.Bench.Repairs.RmsNormTriton.KernelSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

theorem row_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i : Fin c.B,i.val<c.N → pid*c.N+i.val<bounds c.X)
    (weight : ∀ i : Fin c.B,i.val<c.N → i.val<bounds c.W)
    (output : ∀ i : Fin c.B,i.val<c.N → pid*c.N+i.val<bounds c.Y)
    (epsilon : α := ActualRow.argument M eps) :
    TraceSafe (engine M) bounds (ActualRow.source c) (ActualRow.entry epsilon pid s) := by
  simp [TraceSafe,ActualRow.source,runtimeKernel,ActualRow.entry,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,bop,natLt,rowState,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  exact ⟨input,weight,output⟩

theorem schedule_runs {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (eps : ℝ)
    (rows : List Nat) (s : State α) (epsilon : α := ActualRow.argument M eps) :
    runSchedule (rows.map (fun pid => invocation M c eps pid epsilon)) s =
      ActualGrid.runRows M c epsilon rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,ActualGrid.runRows,invocation,Invocation.run,
      Invocation.initial,bindRegisters,List.foldl_nil,Option.getD_none,ActualGrid.rowExec,ActualRow.entry,
      rowState,FP.Structural.exec,List.foldl_cons]
    congr 1
    funext t
    exact ih t

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (eps : ℝ)
    (rows : List Nat) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ pid ∈ rows,∀ i : Fin c.B,i.val<c.N → pid*c.N+i.val<bounds c.X)
    (weight : ∀ i : Fin c.B,i.val<c.N → i.val<bounds c.W)
    (output : ∀ pid ∈ rows,∀ i : Fin c.B,i.val<c.N → pid*c.N+i.val<bounds c.Y)
    (epsilon : α := ActualRow.argument M eps) :
    ScheduleSafe bounds (rows.map (fun pid => invocation M c eps pid epsilon)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨row_safe M c eps pid s bounds (input pid (by simp)) weight (output pid (by simp)) epsilon,?_⟩
    intro t _
    exact ih t (fun p hp => input p (by simp [hp])) (fun p hp => output p (by simp [hp]))

#axiomsClean row_safe
#axiomsClean schedule_runs
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.RmsNormTriton.KernelSafety
