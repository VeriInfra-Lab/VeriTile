import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.FPExpandDimsPrecision
open VeriTile.Triton FP.Structural

/-- Both operands acquire their precision before a shape-only view. There is
no other fp32 operand around the multiply to conceal lost view metadata. -/
def source (dtype : FloatDType) (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = DTYPE, Y = DTYPE
  i = tl.arange(0, 2)
  j = tl.arange(0, 3)
  x = tl.load(X + i).to(tl.float32)
  y = tl.load(Y + j).to(tl.float32)
  product = x[:, None] * y[None, :]
}

theorem expanded_multiply_is_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (s : State α) :
    ∃ t, FP.Structural.exec M (source .fp32 X Y) s = some t ∧
      t.regs .real [2,3] "product" = some (fun i : TileIndex [2,3] =>
        M.binary (some .fp32) .real .mul
          (M.cast (some .fp32) .fp32 .real ((s.mem X i.1.val).read .fp32))
          (M.cast (some .fp32) .fp32 .real ((s.mem Y i.2.1.val).read .fp32))) := by
  refine ⟨?_, ?_, ?_⟩
  rotate_left
  · simp [source, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, evalOffsets, State.setReg, ofFloat, toFloat, numeric, bop,
      FloatDType.toTileDType, FloatDType.numericDType, ComputeDType.eraseDType]
    rfl
  · simp [State.setReg, bop, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex]
    rfl

#axiomsClean expanded_multiply_is_fp32
end VeriTile.Tests.FPExpandDimsPrecision
