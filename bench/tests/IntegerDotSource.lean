import VeriTile.Triton
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerDotSource
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Explicit int32 result wrap materializes the native accumulator's width.
The small tile here is a semantic regression, not a native launch profile. -/
def source (A B : RegionName) : ComputeKernel := triton {
  tl.region A = tl.int8, B = tl.int8
  i = tl.arange(0, 1)
  a = tl.load(A + i[:, None])
  b = tl.load(B + i[None, :])
  accumulator = tl.zeros([1, 1], dtype=tl.int32)
  accumulator = (accumulator + tl.dot(a, b, out_dtype=tl.int32)).to(tl.int32)
}

noncomputable def initial {α : Type} [Inhabited α] : State α where
  mem := fun r _ => .mk .int (if r = "A" then -2 else -3)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

attribute [local simp 2000] FP.Structural.eval_intToSigned
attribute [local irreducible] FP.IntegerDot.backend

/-- Two negative source int8 values reach the concrete dot adapter and the
explicit int32 accumulator wrap. A Nat-clamping interpretation would give 0. -/
theorem source_signed_product {α : Type} [Inhabited α] (rest : Algebra α) :
    ∃ t, FP.Structural.exec (FP.IntegerDot.backend rest) (source "A" "B") initial = some t ∧
      t.regs .int [1,1] "accumulator" = some (fun _ => (6 : Int)) := by
  classical
  simp (config := { maxSteps := 1000000 }) [source, initial, FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalOp_unfold, evalComputeOp, State.setReg,
    FP.IntegerDot.backend, FP.IntegerDot.supported, FP.IntegerDot.valid, FP.IntegerDot.signed8,
    FP.IntegerDot.wordDot, numeric, numericLt, natLt, bop, integralMod,
    Region.cast, Cell.read, List.finRange_succ, Fin.forall_fin_succ, Fin.isValue,
    IntegerExpr.sameShape, Option.map, Option.bind, FP.Structural.eval_intToSigned]

#axiomsClean source_signed_product
end VeriTile.Tests.IntegerDotSource
