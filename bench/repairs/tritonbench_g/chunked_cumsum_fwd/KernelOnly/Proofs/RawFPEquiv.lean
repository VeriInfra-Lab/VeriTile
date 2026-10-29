import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelObserved
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelNumerics

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.RawNative
open VeriTile Triton FP.Structural
open Kernel Kernel.Scalar
noncomputable section

/-- Raw FP32 kernel execution, both independent native recurrences and the
complete raw memory frame. Existing elementary scalar model laws are retained;
they supply neither source execution nor output values. -/
theorem kernel_fp32
    (c : Geometry) (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords (BitVec 32)) (s : State (Term (BitVec 32))) (read : InputGraph s a)
    (hasBias reverseBias useSoftplus : Bool)
    (target : Numeric.Target) (unary : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (scalar : Numeric.ScalarLaws target unary ports) (lower upper : BitVec 32) :
    let actual := Numeric.nativeModel ports reverseBias
    let reference := Numeric.referenceModel target unary reverseBias
    ∃ t, Kernel.launch (graphAlgebra actual c plan) c hasBias useSoftplus
        (.input lower) (.input upper) s = some t ∧
      LogicalOutputs reference c plan a hasBias useSoftplus lower upper t ∧
      Kernel.Frame c c.grid s t := by
  dsimp only
  obtain ⟨t, ran, values, frame⟩ := Scalar.complete_from (Numeric.nativeModel ports reverseBias)
    c sizes plan a s read hasBias useSoftplus lower upper
  refine ⟨t, ran, ?_, frame⟩
  simpa only [Numeric.model_eq target unary ports scalar reverseBias] using values

#axiomsClean kernel_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.RawNative
