import bench.tritonbench_g.matmul_leakyrelu.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- The complete original source grid implements the full logical matrix sum
and optional leaky activation under the original separation and schedule. -/
specification matmul_leakyrelu_real (p : KernelInterface.Parameters)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) (order : Schedule p.request) :
    VeriTile.Spec.Real (ReaderIO.view p order ⊨ fun xs r c =>
      let value := ∑ k : Fin p.inner, xs.A r k * xs.B k c
      if p.activation == "leaky_relu" then
        if 0 ≤ value then value else (1 / 100 : ℝ) * value
      else value) :=
  ReaderProof.correct p separateA separateB order

#axiomsClean matmul_leakyrelu_real
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
