import bench.repairs.tritonbench_g.attention_kernel_aligned.KernelOnly.Inputs
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.KernelContract
abbrev Call := KernelOnly.Call
abbrev Tensor := KernelOnly.Tensor
abbrev TensorLaws := KernelOnly.TensorLaws
abbrev InputContract := KernelOnly.InputContract
noncomputable abbrev check := KernelOnly.check
abbrev accepted_iff := KernelOnly.check_iff
abbrev kernel_access := KernelOnly.kernel_access
#axiomsClean accepted_iff
#axiomsClean kernel_access
#specNonCircular InputContract avoiding [VeriTile.Triton.FP.Structural.exec,Whole.runPrograms]
end VeriTile.Bench.Repairs.AttentionKernelAligned.KernelContract
