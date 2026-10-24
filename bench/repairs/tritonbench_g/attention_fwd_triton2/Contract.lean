import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Inputs

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelContract
abbrev Call := KernelOnly.Call
abbrev Tensor := KernelOnly.Tensor
abbrev TensorLaws := KernelOnly.TensorLaws
abbrev InputContract := KernelOnly.InputContract
noncomputable abbrev check := KernelOnly.check
abbrev accepted_iff := KernelOnly.accepted_iff
#axiomsClean accepted_iff
#specNonCircular InputContract avoiding [VeriTile.Triton.FP.Structural.exec,Whole001.Whole.runPrograms]
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelContract
