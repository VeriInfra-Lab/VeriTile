import bench.repairs.tritonbench_g.sin_computation.IO
import bench.repairs.tritonbench_g.sin_computation.Contract
import bench.repairs.tritonbench_g.sin_computation.Proofs.Projection
import bench.tritonbench_g.sin_computation.Proofs.Real
import VeriTile.Meta.StatementAudit


namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.ReaderProof
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The original source projection, access safety, region result and exact
frame, assembled outside the logical reader without changing its judgment. -/
theorem correct (X O : RegionName) (n : Nat) :
    Spec.Real (Correct.sinIO X O n ⊨ fun xs i => Real.sin (xs i)) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact VeriTile.Bench.TritonBenchG.SinComputation.sin_kernel_flattenOk X O n blockSize
  · intro bounds s hx ho _
    exact VeriTile.Bench.TritonBenchG.SinComputation.sin_kernel_traceSafe _ _ n blockSize bounds s hx ho
  · intro s xs hx
    obtain ⟨t,he,hv,hf⟩ := VeriTile.Bench.TritonBenchG.SinComputation.sin_kernel_region_run
      X O n blockSize s xs hx
    exact ⟨t,he,hv,fun r a outside _ => hf r a outside⟩

#axiomsClean correct
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.ReaderProof
