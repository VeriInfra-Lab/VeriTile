import bench.repairs.tritonbench_g.uniform_sampling.Correct
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Contract

/-! Optional Tensor/check applications; the primary kernel specifications
remain in Correct.lean and FPEquiv.lean, which do not import this bridge. -/
namespace VeriTile.Bench.Repairs.UniformSampling.TensorBridge
open VeriTile.Triton FP.Structural KernelExecution KernelMath
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section

theorem python_checks_iff_preconditions (output : VeriTile.Torch.TensorMetadata)
    (generator : GeneratorArgument) (lowerIsFloat upperIsFloat : Bool) :
    checkInput output generator lowerIsFloat upperIsFloat=.ok () ↔
      InputContract output generator lowerIsFloat upperIsFloat := checkInput_iff output generator lowerIsFloat upperIsFloat

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.TensorBridge
