import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Source

/-! Inspect the actual full source AST before choosing its execution normal
form. In particular the cdiv-minus-one start must be signed at T=0, and the
boundary load must retain unspecified padding.
-/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.InspectSource
open VeriTile.Triton TypedSource

#reduce fun (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) =>
  (original dtype out_dtype s o T BT).surfaceBody[2]?

end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.InspectSource
