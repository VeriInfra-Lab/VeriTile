import bench.tritonbench_g.relu_triton_kernel.IO
import bench.tritonbench_g.relu_triton_kernel.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ReluTritonKernel.RealIO
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Assemble the complete existing logical IO judgment from source safety,
execution, active-lane values and the exterior frame. The interface is unchanged. -/
theorem implements (N : Nat) : io N ⊨ fun xs i => max 0 (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact KernelReal.flattenOk N
  · intro bounds s hx hout _
    exact KernelReal.traceSafe N bounds s hx hout
  · intro s xs hx
    obtain ⟨t, he, hv, hf⟩ := KernelReal.region_run N s xs hx
    exact ⟨t, he, hv, fun r o outside _ => hf r o outside⟩

end
end VeriTile.Bench.Repairs.ReluTritonKernel.RealIO
