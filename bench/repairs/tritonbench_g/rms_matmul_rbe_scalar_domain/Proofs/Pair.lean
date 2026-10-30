import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.PairIO
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOResults
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Pair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

private def candidate (fp8 outputFP8 : Bool) (p : Arguments) : ComputeKernel :=
  TypedSource.indexCommuted fp8 outputFP8 {p with BM:=16,BN:=64,BK:=64}

private def row (commuted : Bool) : ComputeStmt :=
  .assign .nat [16] "offs_m" (.alg (.add .nat .scalarL
    (if commuted then .mul .nat .nil (.constNat 16) (.ref .nat [] "pid_m")
      else .mul .nat .nil (.ref .nat [] "pid_m") (.constNat 16)) (.arange 16)))

private theorem split_source (fp8 outputFP8 : Bool) (p : Arguments) :
    (source fp8 outputFP8 p).surfaceBody =
      (source fp8 outputFP8 p).surfaceBody.take 17 ++
        row Bool.false :: (source fp8 outputFP8 p).surfaceBody.drop 18 := by
  cases fp8 <;> cases outputFP8 <;> rfl

private theorem split_candidate (fp8 outputFP8 : Bool) (p : Arguments) :
    (candidate fp8 outputFP8 p).surfaceBody =
      (source fp8 outputFP8 p).surfaceBody.take 17 ++
        row Bool.true :: (source fp8 outputFP8 p).surfaceBody.drop 18 := by
  cases fp8 <;> cases outputFP8 <;> rfl

private theorem row_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    step M (row Bool.true) s = step M (row Bool.false) s := by
  cases h : s.regs .nat [] "pid_m" <;>
    simp [row,step,evalExpr,evalComputeOp,evalOp_unfold,h,bop,numeric,Option.bind,Nat.mul_comm]
  congr 1
  funext i
  simp [bop,Nat.mul_comm]

private theorem row_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.stepSafe M bounds (row Bool.true) s ↔
      Safety.stepSafe M bounds (row Bool.false) s := by
  simp [row,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

/-- Complete callee equality, including failure and every register/memory cell.
Only a natural-number multiplication is commuted; no floating rule is used. -/
theorem callee_run {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    run M (candidate fp8 outputFP8 p).surfaceBody s =
      run M (source fp8 outputFP8 p).surfaceBody s := by
  conv_lhs => rw [split_candidate]
  conv_rhs => rw [split_source]
  rw [run_append,run_append]
  simp only [run,row_step]

theorem callee_safe {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (candidate fp8 outputFP8 p).surfaceBody s ↔
      Safety.traceSafeList M bounds (source fp8 outputFP8 p).surfaceBody s := by
  conv_lhs => rw [split_candidate]
  conv_rhs => rw [split_source]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  simp only [Safety.traceSafeList_cons,row_safe,row_step]

theorem qkv_run {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 : Bool) (p : Fin 3 → Arguments) (s : State α) :
    run M (code fp8 p) s = run M (qkvCode Bool.false fp8 p) s := by
  change run M (((candidate fp8 Bool.false (qkvArgument p 0)).surfaceBody ++
    (candidate fp8 fp8 (qkvArgument p 1)).surfaceBody) ++
    (candidate fp8 fp8 (qkvArgument p 2)).surfaceBody) s = _
  simp only [qkvCode,calleeCode,Bool.false_eq_true,if_false,run_append]
  simp only [callee_run]

theorem qkv_safe {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 : Bool) (p : Fin 3 → Arguments) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (code fp8 p) s ↔
      Safety.traceSafeList M bounds (qkvCode Bool.false fp8 p) s := by
  change Safety.traceSafeList M bounds (((candidate fp8 Bool.false (qkvArgument p 0)).surfaceBody ++
    (candidate fp8 fp8 (qkvArgument p 1)).surfaceBody) ++
    (candidate fp8 fp8 (qkvArgument p 2)).surfaceBody) s ↔ _
  simp only [qkvCode,calleeCode,Bool.false_eq_true,if_false,
    Safety.traceSafeList_append,run_append]
  simp only [callee_safe,callee_run]

theorem job_run {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (pid : Nat → Nat) (s : State α) :
    (job a fp8 p epsilon theta pid).run s =
      (IO.job a Bool.false fp8 p epsilon theta pid).run s :=
  qkv_run a.model fp8 p _

theorem job_safe {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (pid : Nat → Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (job a fp8 p epsilon theta pid).Safe bounds s ↔
      (IO.job a Bool.false fp8 p epsilon theta pid).Safe bounds s :=
  qkv_safe a.model fp8 p bounds _

theorem schedule_run {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat)) (s : State α) :
    runSchedule (order.map (job a fp8 p epsilon theta)) s =
      runSchedule (order.map (IO.job a Bool.false fp8 p epsilon theta)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,runSchedule,job_run,ih]

theorem schedule_safe {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (order.map (job a fp8 p epsilon theta)) s ↔
      ScheduleSafe bounds (order.map (IO.job a Bool.false fp8 p epsilon theta)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,ScheduleSafe,job_safe,job_run,ih]

theorem equivalent {α : Type} [Inhabited α] (a : Arithmetic α) (zero : ZeroLaws a)
    (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    (IO.qkvIO a Bool.false fp8 p epsilon theta order).EquivalentOn
      (variant a fp8 p epsilon theta order) (IO.InputDomain fp8 p) := by
  intro vals domain bounds bound initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    IOResults.implements a zero Bool.false fp8 p epsilon theta order even apart vals domain initial binding
  refine ⟨safe bounds bound,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe a fp8 p epsilon theta order bounds initial).mpr (safe bounds bound)
  · exact (schedule_run a fp8 p epsilon theta order initial).trans ran
  · intro o j active
    rfl

#axiomsClean equivalent
#axiomsClean callee_run
#axiomsClean qkv_run
#axiomsClean schedule_run
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Pair
