import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.RawStringDispatch
open VeriTile.Triton FP.Structural

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["kind"]

def source (X Y : RegionName) (kind : String) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    if kind == "square":
        value = x * x
    elif kind == "shift":
        value = x + 1.0
    else:
        value = x
    tl.store(Y, value)
  }

/-- Every string takes its actual branch; the selected floating operations
remain opaque, including distinct literal and load conversions. -/
theorem dispatch {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (kind : String) (s : State α) :
    ∃ final, exec M (source X Y kind) s = some final ∧
      final.mem Y 0 = Cell.mk .real
        (let x := M.fp32Load ((s.mem X 0).read .real)
         if kind = "square" then M.binary (some .fp32) .real .mul x x
         else if kind = "shift" then
           M.binary (some .fp32) .real .add x (M.literal (some .fp32) .real 1.0)
         else x) := by
  by_cases square : kind = "square"
  · simp [source,square,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,bop,numeric,
      store,State.setReg,State.write,TileShape.allIndices,Region.cast]
  · by_cases shift : kind = "shift" <;>
      simp [source,square,shift,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
        evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,bop,numeric,
        store,State.setReg,State.write,TileShape.allIndices,Region.cast]

#axiomsClean dispatch
end VeriTile.Tests.RawStringDispatch
