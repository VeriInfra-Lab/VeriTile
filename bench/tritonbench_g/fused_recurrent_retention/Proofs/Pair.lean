import bench.tritonbench_g.fused_recurrent_retention.Proofs.PublicResult
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Pair
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def forward (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) : ComputeKernel :=
  Surface.forwardCompareRight .fp32 .fp32 r.q r.k r.v r.o r.initial r.final
    p.qHead p.qTime p.qChannel p.vHead p.vTime p.vChannel p.B p.H p.T 0
    p.BK p.BV p.DK p.DV hasSeed storeFinal

def backward (p : Layout) (r : BackwardRegions) (hasSeed hasFinal : Bool) : ComputeKernel :=
  Surface.backwardCompareRight .fp32 .fp32 r.q r.k r.v r.do_ r.dq r.dk r.dv r.initial r.finalGrad
    p.doBatch p.doHead p.doTime p.doChannel p.dfBatch p.dfHead p.dfKey p.dfValue
    p.qHead p.qTime p.qChannel p.vHead p.vTime p.vChannel p.B p.H p.T 0
    p.BK p.BV p.DK p.DV hasSeed hasFinal

def kernel (right : Bool) (mode : Mode) (p : Layout) (seed final : Bool) : ComputeKernel :=
  match right,mode with
  | Bool.false,.forward => Program.forward p forwardRegions seed final
  | Bool.false,.backward => Program.backward p backwardRegions seed final
  | Bool.true,.forward => forward p forwardRegions seed final
  | Bool.true,.backward => backward p backwardRegions seed final

private def prefixLength (mode : Mode) : Nat := match mode with | .forward => 9 | .backward => 10

private def mask (right : Bool) (p : Layout) : ComputeStmt :=
  let index : Op .nat [p.BK] := .add .nat .scalarL
    (.mul .nat .nil (.ref .nat [] "i_k") (.constNat p.BK)) (.arange p.BK)
  .assign .bool [p.BK] "mask_bk" (.alg
    (if right then .gt .nat .scalarL (.constNat p.DK) index
      else .lt .nat .scalarR index (.constNat p.DK)))

private theorem split_source (mode : Mode) (p : Layout) (seed final : Bool) :
    (kernel Bool.false mode p seed final).surfaceBody =
      (kernel Bool.false mode p seed final).surfaceBody.take (prefixLength mode) ++
        mask Bool.false p :: (kernel Bool.false mode p seed final).surfaceBody.drop (prefixLength mode+1) := by
  cases mode <;> rfl

private theorem split_candidate (mode : Mode) (p : Layout) (seed final : Bool) :
    (kernel Bool.true mode p seed final).surfaceBody =
      (kernel Bool.false mode p seed final).surfaceBody.take (prefixLength mode) ++
        mask Bool.true p :: (kernel Bool.false mode p seed final).surfaceBody.drop (prefixLength mode+1) := by
  cases mode <;> rfl

private theorem mask_run {α : Type} [Inhabited α] (M : Algebra α) (p : Layout) (s : State α) :
    step M (mask Bool.true p) s = step M (mask Bool.false p) s := by
  cases h : s.regs .nat [] "i_k" <;>
    simp [mask,step,evalExpr,evalComputeOp,evalOp_unfold,h,bop,numeric,numericLt,natLt,Option.bind]
  congr 1

private theorem mask_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Layout)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.stepSafe M bounds (mask Bool.true p) s ↔
      Safety.stepSafe M bounds (mask Bool.false p) s := by
  simp [mask,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

/-- Complete forward/reverse kernel state equality, including every loop. -/
theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (mode : Mode)
    (p : Layout) (seed final : Bool) (s : State α) :
    run M (kernel Bool.true mode p seed final).surfaceBody s =
      run M (kernel Bool.false mode p seed final).surfaceBody s := by
  conv_lhs => rw [split_candidate]
  conv_rhs => rw [split_source]
  rw [run_append,run_append]
  simp only [run,mask_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (mode : Mode)
    (p : Layout) (seed final : Bool) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (kernel Bool.true mode p seed final).surfaceBody s ↔
      Safety.traceSafeList M bounds (kernel Bool.false mode p seed final).surfaceBody s := by
  conv_lhs => rw [split_candidate]
  conv_rhs => rw [split_source]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  simp only [Safety.traceSafeList_cons,mask_safe,mask_run]

def job {α : Type} (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) (pid : ProgramId) :
    Invocation (Staged.Node α) :=
  {PublicData.job mode a s p incoming finalGradient seed final pid with
    kernel := kernel Bool.true mode (KernelInterface.layout s incoming finalGradient) seed final}

private theorem original_kernel (mode : Mode) (s : Shape)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    kernel Bool.false mode (KernelInterface.layout s incoming finalGradient) seed final =
      Quoted.kernel mode s (KernelInterface.buffers s incoming finalGradient) seed final := by
  cases mode <;> rfl

theorem invocation_run {α : Type} [Inhabited α] (mode : Mode) (a : Arithmetic α)
    (s : Shape) (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat)
    (seed final : Bool) (pid : ProgramId) (u : State (Staged.Node α)) :
    (job mode a s p incoming finalGradient seed final pid).run u =
      (PublicData.job mode a s p incoming finalGradient seed final pid).run u := by
  change run (Staged.algebra a p.plans)
    (kernel Bool.true mode (KernelInterface.layout s incoming finalGradient) seed final).surfaceBody
    (prepare a s pid u) = run (Staged.algebra a p.plans)
    (Quoted.kernel mode s (KernelInterface.buffers s incoming finalGradient) seed final).surfaceBody
    (prepare a s pid u)
  rw [kernel_run,original_kernel]

theorem invocation_safe {α : Type} [Inhabited α] (mode : Mode) (a : Arithmetic α)
    (s : Shape) (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat)
    (seed final : Bool) (pid : ProgramId) (bounds : RegionName → Nat) (u : State (Staged.Node α)) :
    (job mode a s p incoming finalGradient seed final pid).Safe bounds u ↔
      (PublicData.job mode a s p incoming finalGradient seed final pid).Safe bounds u := by
  change Safety.traceSafeList (Staged.algebra a p.plans) bounds
    (kernel Bool.true mode (KernelInterface.layout s incoming finalGradient) seed final).surfaceBody
    (prepare a s pid u) ↔ Safety.traceSafeList (Staged.algebra a p.plans) bounds
    (Quoted.kernel mode s (KernelInterface.buffers s incoming finalGradient) seed final).surfaceBody
    (prepare a s pid u)
  rw [kernel_safe,original_kernel]

theorem mapped_run {α : Type} [Inhabited α] (mode : Mode) (a : Arithmetic α)
    (s : Shape) (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat)
    (seed final : Bool) (pids : List ProgramId) (u : State (Staged.Node α)) :
    runSchedule (pids.map (job mode a s p incoming finalGradient seed final)) u =
      runSchedule (pids.map (PublicData.job mode a s p incoming finalGradient seed final)) u := by
  induction pids generalizing u with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,runSchedule,invocation_run,ih]

theorem mapped_safe {α : Type} [Inhabited α] (mode : Mode) (a : Arithmetic α)
    (s : Shape) (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat)
    (seed final : Bool) (pids : List ProgramId) (bounds : RegionName → Nat) (u : State (Staged.Node α)) :
    ScheduleSafe bounds (pids.map (job mode a s p incoming finalGradient seed final)) u ↔
      ScheduleSafe bounds (pids.map (PublicData.job mode a s p incoming finalGradient seed final)) u := by
  induction pids generalizing u with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run,ih]

def variant {α : Type} (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    ProgramIO.Variant (PublicData.io mode a s p incoming finalGradient seed final) where
  schedule := fun _ => (gridPrograms s).map (job mode a s p incoming finalGradient seed final)
  observe := (PublicData.io mode a s p incoming finalGradient seed final).observe

def original {α : Type} (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :=
  ProgramIO.Implementation.original (PublicData.io mode a s p incoming finalGradient seed final) (fun _ => True)

def compareRight {α : Type} (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :=
  ProgramIO.Implementation.ofVariant (variant mode a s p incoming finalGradient seed final) (fun _ => True)

/-- Both complete original mode grids, with exact tagged output cells and both
exterior frames. No floating numerical law is used by this transformation. -/
theorem equivalent {α : Type} [Inhabited α] (mode : Mode) (a : Arithmetic α)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (PublicData.io mode a s p incoming finalGradient seed final).Equivalent
      (variant mode a s p incoming finalGradient seed final) := by
  intro values domain bounds bound initial binding
  obtain ⟨safe,result,ran,outputs,frame⟩ :=
    PublicData.complete mode a s shape p incoming finalGradient seed final
      values domain bounds bound initial binding
  refine ⟨safe,?_,result,result,ran,?_,?_,frame,frame⟩
  · exact (mapped_safe mode a s p incoming finalGradient seed final (gridPrograms s) bounds initial).mpr safe
  · exact (mapped_run mode a s p incoming finalGradient seed final (gridPrograms s) initial).trans ran
  · intro o j active
    rfl

#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Pair
