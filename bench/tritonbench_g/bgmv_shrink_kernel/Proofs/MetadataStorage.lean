import bench.tritonbench_g.bgmv_shrink_kernel.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Pasted
noncomputable section
/-- Both native metadata widths widen to the same signed64 source channel.
This is equality of complete kernels, for every runtime argument, floating
storage type, branch and loop, before any invocation or numerical premise. -/
theorem original_metadata_widths (input_dtype output_dtype : VeriTile.Triton.FloatDType) :
    originalKernel input_dtype output_dtype .i32 = originalKernel input_dtype output_dtype .i64 := by rfl

theorem candidate_metadata_widths (input_dtype output_dtype : VeriTile.Triton.FloatDType) :
    candidateKernel input_dtype output_dtype .i32 = candidateKernel input_dtype output_dtype .i64 := by rfl

#axiomsClean original_metadata_widths
#axiomsClean candidate_metadata_widths
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Pasted
