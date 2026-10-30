import bench.tritonbench_g.attention_kernel.Proofs.Inputs

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelContract
abbrev Call := KernelOnly.Call
abbrev Tensor := KernelOnly.Tensor
abbrev TensorLaws := KernelOnly.TensorLaws
abbrev InputContract := KernelOnly.InputContract
noncomputable abbrev check := KernelOnly.check
abbrev accepted_iff := KernelOnly.check_iff
#axiomsClean accepted_iff
#specNonCircular InputContract avoiding [VeriTile.Triton.FP.Structural.exec,Whole001.Whole.runPrograms]
end VeriTile.Bench.Repairs.AttentionKernel.KernelContract
