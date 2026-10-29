import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Semantics.Integer
import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Float.Integer
import VeriTile.Meta.StatementAudit

/-! Runtime integer `//` and `%` use C truncation. The tests exercise the
frontend, both execution interfaces, signs, broadcast and missing registers.
Python constexpr arithmetic is supplied explicitly through Lean antiquotation.
All integers here are mathematical Int; hardware overflow and zero division
are outside the claimed source correspondence. No floating rule is assumed. -/
namespace RuntimeIntegerDivisionTests
open VeriTile.Triton

def runtime (x y : Int) : ComputeKernel := triton {
  numerator = $((x : Int))
  denominator = $((y : Int))
  quotient = numerator // denominator
  remainder = numerator % denominator
}

theorem runtime_lowering (x y : Int) : (runtime x y).surfaceBody = [
    .assign .int [] "numerator" (.alg (.constInt x)),
    .assign .int [] "denominator" (.alg (.constInt y)),
    .assign .int [] "quotient" (.alg (Op.truncDiv .nil
      (.ref .int [] "numerator") (.ref .int [] "denominator"))),
    .assign .int [] "remainder" (.alg (Op.truncMod .nil
      (.ref .int [] "numerator") (.ref .int [] "denominator")))] := rfl

noncomputable def realResults (k : ComputeKernel) (s : BlockState) :=
  (k.toAlgorithm?.toOption.bind (fun a => VeriTile.Triton.exec a s)).map
    (fun t => (t.regs .int [] "quotient", t.regs .int [] "remainder"))

theorem real_runtime (x y : Int) (s : BlockState) :
    realResults (runtime x y) s =
      some (some (Tile.scalar (x.tdiv y)), some (Tile.scalar (x.tmod y))) := by
  simp only [realResults, runtime, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, bind, Option.bind_some, VeriTile.Triton.exec]
  simp only [stepStmts, stepStmt, eval_truncDiv, eval_truncMod]
  simp [VeriTile.Triton.evalOp, BlockState.setReg, Tile.bop, Tile.scalar, -Tile.scalar_eta]

noncomputable def fpResults {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (k : ComputeKernel) (s : FP.Structural.State α) :=
  (FP.Structural.exec M k s).map
    (fun t => (t.regs .int [] "quotient", t.regs .int [] "remainder"))

theorem fp_runtime {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (x y : Int) (s : FP.Structural.State α) :
    fpResults M (runtime x y) s =
      some (some (fun _ => x.tdiv y), some (fun _ => x.tmod y)) := by
  simp only [fpResults, runtime, FP.Structural.exec, FP.Structural.run,
    FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.eval_truncDiv, FP.Structural.eval_truncMod]
  simp [FP.Structural.evalOp, FP.Structural.State.setReg]
  constructor <;> rfl

theorem negative_dividend (s : BlockState) :
    realResults (runtime (-7) 3) s =
      some (some (Tile.scalar (-2)), some (Tile.scalar (-1))) := by
  exact real_runtime (-7) 3 s

theorem negative_divisor (s : BlockState) :
    realResults (runtime 7 (-3)) s =
      some (some (Tile.scalar (-2)), some (Tile.scalar 1)) := by
  exact real_runtime 7 (-3) s

theorem both_negative (s : BlockState) :
    realResults (runtime (-7) (-3)) s =
      some (some (Tile.scalar 2), some (Tile.scalar (-1))) := by
  exact real_runtime (-7) (-3) s

theorem exact_multiple (s : BlockState) :
    realResults (runtime (-6) 3) s =
      some (some (Tile.scalar (-2)), some (Tile.scalar 0)) := by
  exact real_runtime (-6) 3 s

def constexpr : ComputeKernel := triton {
  quotient = $((Int.fdiv (-7) 3 : Int))
  remainder = $((Int.fmod (-7) 3 : Int))
}

theorem constexpr_floor_lowering : constexpr.surfaceBody = [
    .assign .int [] "quotient" (.alg (.constInt (-3))),
    .assign .int [] "remainder" (.alg (.constInt 2))] := rfl

-- Signed context propagates through the quotient to the subtraction.
def windowBound : ComputeKernel := triton {
  quotient = ((tl.program_id(0) - $(7)) // $(3)) + $((0 : Int))
}

theorem signed_context_lowering : windowBound.surfaceBody = [
    .assign .int [] "quotient" (.alg (.add .int .nil
      (Op.truncDiv .nil
        (.sub .int .nil (.castNatToInt (.programId 0)) (.constInt 7)) (.constInt 3))
      (.constInt 0)))] := rfl

def natural : ComputeKernel := triton {
  quotient = tl.program_id(0) // $(3)
  remainder = tl.program_id(0) % $(3)
}

theorem natural_lowering_unchanged : natural.surfaceBody = [
    .assign .nat [] "quotient" (.alg (.floorDiv .nat .nil (.programId 0) (.constNat 3))),
    .assign .nat [] "remainder" (.alg (.mod .nat .nil (.programId 0) (.constNat 3)))] := rfl

/-- Existing execution proofs unfold the interpreter directly. Transparent AST
constructors must preserve that path, without needing new integer lemmas. -/
theorem natural_execution_unfolds {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) :
    (FP.Structural.exec M natural s).map
      (fun t => (t.regs .nat [] "quotient", t.regs .nat [] "remainder")) =
      some (some (fun _ => s.pids 0 / 3), some (fun _ => s.pids 0 % 3)) := by
  simp [natural, FP.Structural.exec, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold,
    FP.Structural.State.setReg, FP.Structural.integralFloorDiv, FP.Structural.integralMod]
  constructor <;> rfl

theorem signed_execution_unfolds {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) :
    fpResults M (runtime (-7) 3) s =
      some (some (fun _ => (-2 : Int)), some (fun _ => (-1 : Int))) := by
  simp [fpResults, runtime, FP.Structural.exec, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold, FP.Structural.State.setReg,
    FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.integralFloorDiv,
    FP.Structural.bop]
  funext i
  cases i
  rfl

theorem broadcast_values {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) (s : FP.Structural.State α) (n : Nat) (y : Int)
    (x : FP.Structural.Values α .int [n]) (h : s.regs .int [n] "x" = some x) :
    FP.Structural.evalOp M p
      (Op.truncDiv .scalarR (.ref .int [n] "x") (.constInt y)) s =
        some (fun i => (x i).tdiv y) ∧
    FP.Structural.evalOp M p
      (Op.truncMod .scalarR (.ref .int [n] "x") (.constInt y)) s =
        some (fun i => (x i).tmod y) := by
  rw [FP.Structural.eval_truncDiv, FP.Structural.eval_truncMod]
  simp [FP.Structural.evalOp, h]
  constructor <;> rfl

theorem missing_operand {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) (s : FP.Structural.State α)
    (h : s.regs .int [] "missing" = none) :
    FP.Structural.evalOp M p
      (Op.truncDiv .nil (.ref .int [] "missing") (.constInt 3)) s = none ∧
    FP.Structural.evalOp M p
      (Op.truncMod .nil (.constInt (-7)) (.ref .int [] "missing")) s = none := by
  rw [FP.Structural.eval_truncDiv, FP.Structural.eval_truncMod]
  simp [FP.Structural.evalOp, h]

/-- Signed cdiv follows the actual helper expression, including negative
operands and the interpreter's total zero-divisor convention. -/
def signedCdiv (x y : Int) : ComputeKernel := triton {
  numerator = $((x : Int))
  denominator = $((y : Int))
  quotient = tl.cdiv(numerator, denominator)
}
def signedCdivReference (x y : Int) : ComputeKernel := triton {
  numerator = $((x : Int))
  denominator = $((y : Int))
  quotient = (numerator + (denominator - 1)) // denominator
}
theorem signed_cdiv_source (x y : Int) :
    signedCdiv x y = signedCdivReference x y := rfl

noncomputable def signedCdivResult {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (x y : Int) (s : FP.Structural.State α) :=
  (FP.Structural.exec M (signedCdiv x y) s).map (fun t => t.regs .int [] "quotient")

theorem signed_cdiv_execution {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (x y : Int) (s : FP.Structural.State α) :
    signedCdivResult M x y s = some (some (fun _ => (x + (y - 1)).tdiv y)) := by
  simp only [signedCdivResult,signedCdiv,FP.Structural.exec,FP.Structural.run,
    FP.Structural.step,FP.Structural.evalExpr,FP.Structural.eval_truncDiv]
  simp [FP.Structural.evalOp,FP.Structural.State.setReg,FP.Structural.numeric,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem negative_cdiv_is_source_truncation {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    signedCdivResult M (-6) 4 s = some (some (fun _ => 0)) := by
  simpa using signed_cdiv_execution M (-6) 4 s

def signedCdivLoop (x y : Int) : ComputeKernel := triton {
  stop = tl.cdiv($((x : Int)), $((y : Int)))
  for i in range(tl.cast(0, tl.int64), stop, 1) {
    last = i
  }
}
def signedCdivLoopReference (x y : Int) : ComputeKernel := triton {
  stop = ($((x : Int)) + ($((y : Int)) - 1)) // $((y : Int))
  for i in range(tl.cast(0, tl.int64), stop, 1) {
    last = i
  }
}
/-- Later use as a range bound cannot pin an explicitly signed cdiv to Nat. -/
theorem signed_cdiv_range_bound (x y : Int) :
    signedCdivLoop x y = signedCdivLoopReference x y := rfl

def signedArangeSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("n",.integer)]
  constexpr := ["B"]
def signedArangeMemory (X Y : RegionName) (B n : Nat) : ComputeKernel :=
  with_triton_call signedArangeSignature call_args [n := n] in triton {
    offsets = tl.arange(0, B).to(tl.int64)
    x = tl.load(X + offsets, mask=offsets < n, other=0.0)
    tl.store(Y + offsets, x, mask=offsets < n)
  }
/-- Pointer and mask uses cannot silently retag the explicit INT64 arange
as Nat. This checks the signed instruction itself, not a renamed reference. -/
theorem signed_arange_cast_survives_address_inference (X Y : RegionName) (B n : Nat) :
    (signedArangeMemory X Y B n).surfaceBody.head? = some
      (.assign .int [B] "offsets" (.alg (.castNatToInt (.arange B)))) := rfl
#axiomsClean signed_arange_cast_survives_address_inference

#axiomsClean signed_cdiv_range_bound
#axiomsClean signed_cdiv_source
#axiomsClean signed_cdiv_execution
#axiomsClean negative_cdiv_is_source_truncation

#guard_msgs (drop info) in
#auditModuleAxioms
end RuntimeIntegerDivisionTests
