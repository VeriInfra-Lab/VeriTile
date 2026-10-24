import bench.repairs.tritonbench_g.rms_norm_triton.Kernels
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.Host
import bench.tritonbench_g.rms_norm_triton.Proofs.ObservedHost

namespace VeriTile.Bench.Repairs.RmsNormTriton.HostFPEquiv
open VeriTile.Bench.TritonBenchG.RmsNormTriton
noncomputable section

inductive PrimitiveBoundary where
  | constants | columnConversion | multiply | add | fma | divideFull | sqrtApproxFTZ
  deriving DecidableEq
def primitiveBoundaries : List PrimitiveBoundary := [.constants,.columnConversion,.multiply,.add,.fma,.divideFull,.sqrtApproxFTZ]
abbrev reference := @Observed.reference

def checked (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (model : Observed.NativeModel (BitVec 32)) (profile : Observed.ReductionFamily)
    (rt : Whole.Runtime (Observed.Term (BitVec 32)) input weight n.dimensions (Observed.entryGraph entry) live)
    (setup : Nat → Whole.Prepared (Observed.Term (BitVec 32)) input weight (Observed.entryGraph entry) live →
      Except String (Whole.Schedule input n.dimensions))
    (api : ∀ a : Whole.Prepared (Observed.Term (BitVec 32)) input weight (Observed.entryGraph entry) live,
      PreparedAPI a n.dimensions) (callerDevice : Nat) : Except InputError (Host.Returned (BitVec 32)) :=
  match checkTensorInput x w n eps with
  | .error e => .error e
  | .ok _ => .ok ⟨Observed.executeHost model profile eps.value
      (checkedRuntime x n.dimensions (Host.contextRuntime rt setup) api),callerDevice⟩

theorem wrapper_fp32 (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x w n eps) (binding : Binding x w input weight live)
    (model : Observed.NativeModel (BitVec 32)) (epsilonArgument : model.epsilon=eps.runtimeWord)
    (profile : Observed.ReductionFamily)
    (rt : Whole.Runtime (Observed.Term (BitVec 32)) input weight n.dimensions (Observed.entryGraph entry) live)
    (setup : Nat → Whole.Prepared (Observed.Term (BitVec 32)) input weight (Observed.entryGraph entry) live →
      Except String (Whole.Schedule input n.dimensions))
    (api : ∀ a : Whole.Prepared (Observed.Term (BitVec 32)) input weight (Observed.entryGraph entry) live,
      PreparedAPI a n.dimensions) (callerDevice : Nat) :
    ∃ result,checked x w n eps model profile rt setup api callerDevice=.ok result ∧
      result.currentDevice=callerDevice ∧
      (∀ a : Whole.Prepared (Observed.Term (BitVec 32)) input weight (Observed.entryGraph entry) live,
        a.xc.tensor.device=x.device.index) ∧
      model.epsilon=eps.runtimeWord ∧
      Observed.FiniteHostSpec model profile
        (checkedRuntime x n.dimensions (Host.contextRuntime rt setup) api)
        (runtimeAdmission x w n eps c binding (Host.contextRuntime rt setup) api) result.outcome := by
  refine ⟨⟨Observed.executeHost model profile eps.value
    (checkedRuntime x n.dimensions (Host.contextRuntime rt setup) api),callerDevice⟩,?_,rfl,fun a=>Host.owner binding a,epsilonArgument,?_⟩
  · simp [checked,(checkTensorInput_iff x w n eps).mpr c]
  · exact Observed.finite_host_outcomes model profile eps.value _ (binding.domain n eps c) _

#axiomsClean wrapper_fp32
#print axioms wrapper_fp32
end
end VeriTile.Bench.Repairs.RmsNormTriton.HostFPEquiv
