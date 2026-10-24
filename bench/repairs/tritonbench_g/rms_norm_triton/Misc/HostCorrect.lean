import bench.repairs.tritonbench_g.rms_norm_triton.Kernels
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.RmsNormTriton.HostCorrect
open VeriTile.Bench.TritonBenchG.RmsNormTriton
noncomputable section

theorem wrapper_correct (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x w n eps) (binding : Binding x w input weight live)
    (rt : Whole.Runtime ℝ input weight n.dimensions entry live)
    (setup : Nat → Whole.Prepared ℝ input weight entry live → Except String (Whole.Schedule input n.dimensions))
    (api : ∀ a : Whole.Prepared ℝ input weight entry live,PreparedAPI a n.dimensions) (callerDevice : Nat) :
    ∃ result,Host.checked x w n eps Whole.realAlgebra rt setup api callerDevice=.ok result ∧
      result.currentDevice=callerDevice ∧
      (∀ a : Whole.Prepared ℝ input weight entry live,a.xc.tensor.device=x.device.index) ∧
      Whole.FiniteHostSpec (fun row i => Whole.realFormula (Whole.columns n.dimensions) eps.value
        (fun j => input.read entry (row*Whole.columns n.dimensions+j)) (weight.read entry) i)
        (checkedRuntime x n.dimensions (Host.contextRuntime rt setup) api)
        (runtimeAdmission x w n eps c binding (Host.contextRuntime rt setup) api) result.outcome := by
  refine ⟨⟨Whole.executeHost Whole.realAlgebra eps.value
    (checkedRuntime x n.dimensions (Host.contextRuntime rt setup) api),callerDevice⟩,?_,rfl,fun a=>Host.owner binding a,?_⟩
  · simp [Host.checked,(checkTensorInput_iff x w n eps).mpr c]
  · exact Whole.finite_real_host_outcomes eps.value _ (binding.domain n eps c) _

abbrev rejected := @Host.rejected
#axiomsClean wrapper_correct
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.RmsNormTriton.HostCorrect
