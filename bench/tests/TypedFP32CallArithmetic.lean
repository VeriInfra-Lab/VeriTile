import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.TypedFP32CallArithmetic
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 2000000

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("scale",.fp32)]
  typedFP32Storage := Bool.true

def multiply (X Y : RegionName) (scale : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [scale := scale] in triton {
    value = tl.load(X)
    value *= scale
    tl.store(Y, value)
  }

def ordinary (X Y : RegionName) (scale : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [scale := scale] in triton {
    value = tl.load(X)
    value = value * scale
    tl.store(Y, value)
  }

theorem multiply_typed_store {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (scale : ℝ) (s : State α) :
    ∃ t, exec M (multiply X Y scale) s = some t ∧
      t.mem Y 0 = Cell.mk .fp32 (M.cast none .real .fp32
        (M.binary (some .fp32) .real .mul
          (M.cast (some .fp32) .fp32 .real ((s.mem X 0).read .fp32))
          (M.literal (some .fp32) .real scale))) ∧
      (∀ r o, r ≠ Y ∨ o ≠ 0 → t.mem r o = s.mem r o) := by
  simp [multiply, exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, State.setReg, Option.bind, numeric, bop,
    store, evalOffsets, State.write, TileShape.allIndices,
    FloatDType.numericDType, FloatDType.toTileDType, ComputeDType.eraseDType,
    ofFloat, toFloat, Region.cast]
  intro r o outside same zero
  change r = Y at same
  exact (outside.elim (fun h => h same) (fun h => h zero)).elim

theorem augmented_matches_binary (X Y : RegionName) (scale : ℝ) :
    multiply X Y scale = ordinary X Y scale := rfl

#axiomsClean multiply_typed_store
#axiomsClean augmented_matches_binary
end
end VeriTile.Tests.TypedFP32CallArithmetic
