import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_Erasure
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_RandomDomain

/-! A normal form of the complete source projection. The source equality
below checks all twenty-three statements, including the simultaneous tuple's
five operand captures and four result writes. This is not an input/output
specification; the independent scalar/output formula lives in ScalarSpec and
Formula and is connected by the execution proof.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Algorithm
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def indexOp (swapped : Bool) (B : Nat) : Op .nat [B] :=
  .add .nat .scalarL (if swapped then
    .mul .nat .nil (.ref .nat [] "BLOCK_arg") (.programId 0) else
    .mul .nat .nil (.programId 0) (.ref .nat [] "BLOCK_arg")) (.arange B)

end
end VeriTile.Bench.TritonBenchG.UniformSampling.Algorithm
