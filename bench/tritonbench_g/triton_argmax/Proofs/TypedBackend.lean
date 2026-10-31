import bench.tritonbench_g.triton_argmax.Proofs.TypedReduction
import bench.tritonbench_g.triton_argmax.Proofs.Mathematical_DimBackend
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

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
