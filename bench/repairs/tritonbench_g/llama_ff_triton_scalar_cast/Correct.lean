import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Correct
open VeriTile Triton FP.Structural Execution
open scoped BigOperators LogicalDomainIO
noncomputable section

/-- Complete decoded RMSNorm–SwiGLU on logical row/column coordinates.
The original half/int8 branches, all seven arrays and byte domain are retained. -/
specification kernel_correct (p : Parameters) :
    Spec.Real (ReaderIO.view p ⊨{ReaderIO.byteDomain} fun xs row column =>
      let rho := 1 / Real.sqrt ((∑ k ∈ Finset.range p.inner, xs.arrays.x row.val k ^ 2) /
        (p.inner : ℝ) + p.epsilon)
      let u := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w1 column.val k) * rho
      let v := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w3 column.val k) * rho
      (u * (1 / (1 + Real.exp (-u)))) * v) :=
  ReaderProof.correct p

#axiomsClean kernel_correct
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Correct
