import bench.repairs.tritonbench_g.cosine_compute.IO
import bench.tritonbench_g.cosine_compute.Proofs.Real

namespace VeriTile.Bench.Repairs.CosineCompute.ReaderProof
open VeriTile.Triton
noncomputable section

theorem cosine_kernel_correctness (n : Nat) :
    (IO.cosine n).Implements (fun xs i => Real.cos (xs i)) :=by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact Projection.kernel_flattenOk "a" "b" n blockSize
  · intro bounds s hx ho _
    exact Projection.kernel_traceSafe _ _ n blockSize bounds s hx ho
  · intro s xs hx
    obtain ⟨t,he,hv,hf⟩ := VeriTile.Bench.TritonBenchG.CosineCompute.RealProof.cos_func_region_run
      "a" "b" n blockSize s xs hx
    refine ⟨t,?_,hv,fun r a outside _ => hf r a outside⟩
    exact (Projection.kernel_execution "a" "b" n blockSize s).trans he

end
end VeriTile.Bench.Repairs.CosineCompute.ReaderProof
