import bench.tritonbench_g.rmsnorm_fused_llama.Kernels

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
noncomputable section
open VeriTile.Triton
open VeriTile.Bench.Repairs.RmsnormFusedLlama

def typedKernel (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  (if v then runtimeCandidate X Y W N B stride else runtimeKernel X Y W N B stride).withScalarArguments
    [ScalarArgument.fp32 "eps_arg" eps]

end
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
