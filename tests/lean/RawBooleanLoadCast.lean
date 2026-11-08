import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.RawBooleanLoadCast
open VeriTile.Triton FP.Structural
set_option linter.unusedSimpArgs false

inductive SignedStorage where | i32 | i64

def signature (storage : SignedStorage) : CallSignature where
  pointers := [("mask",match storage with | .i32 => .i32 | .i64 => .i64),("output",.i32)]

def source (storage : SignedStorage) (mask output : RegionName) : ComputeKernel :=
  with_triton_signature (signature storage) in triton {
    enabled = tl.load(mask).to(tl.int1)
    value = tl.where(enabled, tl.cast(1, tl.int32), tl.cast(0, tl.int32))
    tl.store(output, value)
  }

/-- Casting a signed loaded integer to a mask tests nonzero, including
negative and even nonzero values. Both signed ABI widths use the actual
loaded integer channel, with no floating conversion or low-bit mask. -/
theorem nonzero_mask {α : Type} [Inhabited α] (M : Algebra α)
    (storage : SignedStorage) (mask output : RegionName) (v : Int) (s : State α)
    (input : s.mem mask 0 = Cell.mk .int v) :
    ∃ final, exec M (source storage mask output) s = some final ∧
      final.mem output 0 = Cell.mk .int (if v ≠ 0 then 1 else 0) := by
  cases storage <;> by_cases zero : v = 0
  all_goals
    simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,bop,numeric,numericEq,numericLt,natLt,store,Region.cast,
      State.setReg,State.write,TileShape.allIndices,input,zero,integralMod,
      IntegerExpr.sameShape,IntegerExpr.same_left,IntegerExpr.same_right,IntegerExpr.signedBits]

#axiomsClean nonzero_mask
end VeriTile.Tests.RawBooleanLoadCast
