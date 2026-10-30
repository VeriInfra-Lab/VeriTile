import bench.repairs.tritonbench_g.sin_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.sin_kernel.IO
import bench.tritonbench_g.sin_kernel.Proofs.Real
import VeriTile.Meta.StatementAudit


namespace VeriTile.Bench.Repairs.SinKernel.ReaderProof
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The original source projection, access safety, region result and exact
frame, assembled outside the logical reader without changing its judgment. -/
theorem correct (n : Nat) :
    Spec.Real (IO.sinIO n ⊨ fun xs i => Real.sin (xs i)) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact VeriTile.Bench.TritonBenchG.SinKernel.kernel_function_flattenOk "x_ptr" "output_ptr" n blockSize
  · intro bounds s hx ho _
    exact VeriTile.Bench.TritonBenchG.SinKernel.kernel_function_traceSafe _ _ n blockSize bounds s hx ho
  · intro s xs hx
    obtain ⟨t,he,hv,hf⟩ := VeriTile.Bench.TritonBenchG.SinKernel.kernel_function_region_run
      "x_ptr" "output_ptr" n blockSize s xs hx
    exact ⟨t,he,hv,fun r a outside _ => hf r a outside⟩

#axiomsClean correct
end
end VeriTile.Bench.Repairs.SinKernel.ReaderProof
