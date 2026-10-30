import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedReduction
import bench.tritonbench_g.triton_argmax.Proofs.DimBackend
import VeriTile.Triton.Float.Control

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section

/-- Concrete FP32 NaN-sensitive comparison backend. Only sanitized inputs
reach the ordinary numeric indexed max; their ties use the least lane. -/
def backend (rest : Algebra Word) : Algebra Word :=
  { rest with
    negInf := 0xff800000
    fp32Load := id
    compareEq := fun _ _ => some ScalarBinding.eq32
    compareLt := fun _ _ => some (fun x y => ScalarBinding.gt32 y x)
    cast := fun p src dst x => match src,dst with
      | .real,.real | .fp32,.real | .real,.fp32 | .fp32,.fp32 => x
      | _,_ => rest.cast p src dst x
    indexedMax := fun _ dtype => VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed dtype }

/-- Extract a quoted range's body; no hand-written AST is introduced. -/
def rangeBody (code : List ComputeStmt) (index : Nat) : List ComputeStmt :=
  match code[index]? with
  | some (.forRange _ _ _ _ body) | some (.forRangeDyn _ _ _ _ body) => body
  | _ => []

def body (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) :=
  rangeBody (TypedSource.dimension X O M N K BM BN base index64).surfaceBody 7

example (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) :
    (TypedSource.dimension X O M N K BM BN base index64).surfaceBody.length=12 := by rfl

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
