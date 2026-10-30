import bench.tritonbench_g.bgmv_shrink_kernel.Target
import VeriTile.Triton.Float.TermModel
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.ScalarTree
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section
abbrev Plans := FloatDType → FP.Equational.Schedules

/-- Evaluate the existing explicit tree using only a scalar add and literal.
Every padding zero is retained; no associativity or zero identity is used. -/
def eval (add : α → α → α) (zero : α) (xs : Fin n → α) : ReductionTree n → α
  | .input i => xs i
  | .zero => zero
  | .add a b => add (eval add zero xs a) (eval add zero xs b)

def inputs (dtype : FloatDType) (shape : TileShape) (ax : Fin shape.length)
    (keep : Bool) (xs : Values α dtype.toTileDType shape) :
    TileIndex (TileShape.reduceShape shape ax keep) →
      Fin (TileShape.axisDim shape ax) → α :=
  match keep with
  | Bool.false => fun i k => toFloat dtype (xs (TileShape.insertAxisIndex shape ax i k))
  | Bool.true => fun i k => toFloat dtype (xs (TileShape.replaceAxisIndex shape ax i k))

/-- The formerly abstract typed reduction is instantiated with an explicit
tree of source-precision scalar additions. It still receives the full tile. -/
def reduction (M : Algebra α) (plans : Plans) : VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.Reduction α where
  apply := fun p dtype {shape} ax keep xs i => ofFloat dtype
    (eval (M.binary p dtype .add) (M.literal p dtype 0)
      (inputs dtype shape ax keep xs i) (plans dtype p shape ax keep i).tree)


end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.ScalarTree
