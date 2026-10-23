import bench.repairs.tritonbench_g.quant_transpose_kernel.Kernels
import bench.tritonbench_g.quantize_global.Proofs.IntegerSemantics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- The existing straight-line llrint adapter preserves signed64 registers
and executes the actual pointer store. The stored signed64 register is
observed through its low byte at the source's int8 boundary. No new core semantics. -/
def typed (p : Parameters) (version : Bool) : QuantizeIntegerStage.TypedKernel :=
  ⟨kernel version p, QuantizeIntegerStage.byteOutput ⟨p.output⟩,
    QuantizeIntegerStage.outputRegister⟩

abbrev exec {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (version : Bool) (s : State α) : Option (State α) :=
  QuantizeIntegerStage.exec M (some round) (typed p version) s

/-- Successful execution of the complete grouped source. The signed64 callback
is a scalar primitive model, never an assumed result for a tile or Tensor. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (s : State α) :
    {t : State α // exec M round p Bool.false s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := { maxSteps := 1000000 }) [exec, typed, kernel,
    ComputeKernel.surfaceBody, QuantizeIntegerStage.exec, QuantizeIntegerStage.run,
    QuantizeIntegerStage.step, QuantizeIntegerStage.evalExpr, QuantizeIntegerStage.store,
    QuantizeIntegerStage.storedInteger, QuantizeIntegerStage.byteOutput,
    QuantizeIntegerStage.outputRegister, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.natLt, FP.Structural.bop, FP.Structural.integralFloorDiv,
    FP.Structural.integralMod, ComputeDType.eraseDType,
    ofFloat, toFloat, Region.cast, Option.bind]
  try rfl

private abbrev bodyRun {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) :=
  QuantizeIntegerStage.run M (some round) (QuantizeIntegerStage.byteOutput ⟨p.output⟩)
    QuantizeIntegerStage.outputRegister

private theorem run_append {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (xs ys : List ComputeStmt) (s : State α) :
    bodyRun M round p (xs ++ ys) s =
      (bodyRun M round p xs s).bind (bodyRun M round p ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons st xs ih =>
    change (QuantizeIntegerStage.step M (some round)
      (QuantizeIntegerStage.byteOutput ⟨p.output⟩) QuantizeIntegerStage.outputRegister st s).bind
        (fun t => bodyRun M round p (xs ++ ys) t) =
      ((QuantizeIntegerStage.step M (some round)
        (QuantizeIntegerStage.byteOutput ⟨p.output⟩) QuantizeIntegerStage.outputRegister st s).bind
          (bodyRun M round p xs)).bind (bodyRun M round p ys)
    cases QuantizeIntegerStage.step M (some round)
      (QuantizeIntegerStage.byteOutput ⟨p.output⟩) QuantizeIntegerStage.outputRegister st s with
    | none => rfl
    | some next => exact ih next

/-- Source statement 10 is the sole address-product rewrite. This extracts
that statement from the actual quotation; no second AST is maintained. -/
private def addressBody (p : Parameters) (version : Bool) : List ComputeStmt :=
  ((kernel version p).surfaceBody.drop 10).take 1

private theorem address_run {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (s : State α) :
    bodyRun M round p (addressBody p Bool.false) s =
      bodyRun M round p (addressBody p Bool.true) s := by
  cases hr : s.regs .nat [128] "rm" <;>
    cases hc : s.regs .nat [128] "rn" <;>
    simp (config := { maxSteps := 1000000 }) [bodyRun, addressBody, kernel,
      ComputeKernel.surfaceBody, List.drop, List.take,
      QuantizeIntegerStage.run, QuantizeIntegerStage.step, QuantizeIntegerStage.evalExpr,
      FP.Structural.evalOp_unfold, FP.Structural.numeric, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat, Region.cast,
      hr, hc, Option.bind, Nat.mul_comm]
  all_goals try (congr 1; funext i; simp [FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Nat.mul_comm])

private theorem split_source (p : Parameters) (version : Bool) :
    (kernel version p).surfaceBody =
      (kernel version p).surfaceBody.take 10 ++
        (addressBody p version ++ (kernel version p).surfaceBody.drop 11) := by
  cases version <;> rfl

private theorem run_split {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (version : Bool) (s : State α) :
    exec M round p version s =
      (bodyRun M round p ((kernel version p).surfaceBody.take 10) s).bind
        (bodyRun M round p (addressBody p version ++ (kernel version p).surfaceBody.drop 11)) := by
  change bodyRun M round p (kernel version p).surfaceBody s = _
  conv_lhs => rw [split_source p version]
  exact run_append M round p _ _ s

theorem same_run {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (s : State α) :
    exec M round p Bool.false s = exec M round p Bool.true s := by
  have hp : (kernel Bool.false p).surfaceBody.take 10 =
      (kernel Bool.true p).surfaceBody.take 10 := rfl
  have ht : (kernel Bool.false p).surfaceBody.drop 11 =
      (kernel Bool.true p).surfaceBody.drop 11 := rfl
  have hs : bodyRun M round p (addressBody p Bool.false ++ (kernel Bool.false p).surfaceBody.drop 11) =
      bodyRun M round p (addressBody p Bool.true ++ (kernel Bool.true p).surfaceBody.drop 11) := by
    funext next
    rw [run_append M round p (addressBody p Bool.false) _ next,
      run_append M round p (addressBody p Bool.true) _ next, ht, address_run]
  rw [run_split M round p Bool.false s, run_split M round p Bool.true s, hp, hs]

#axiomsClean execution
#axiomsClean same_run
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelExecution
