import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Kernels
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelContract
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelRealTensor
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelBytes
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelNumerics
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelPlans
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.FPEquiv

import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Proofs.RawCorrect
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Proofs.RawFPEquiv

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.TensorBridge.Real
open VeriTile Triton
open Kernel Kernel.Scalar Kernel.Tensor
noncomputable section

/-- Complete source execution, both independent mathematical formulas in
actual Tensor views, one backing for both outputs, and caller-memory frame.
Metadata bounds are derived from the unchanged checked-input contract. -/
theorem repaired_real
    (c : Geometry) (f : Buffers c hasBias)
    (checked : InputContract f.inputs.dt.metadata f.inputs.rate.metadata
      (f.inputs.bias.map (·.metadata)) (Int.ofNat c.chunk) integerChunk booleanSoftplus floatLimits)
    (plan : PrefixPlan c.BH c.BC) (reverseBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (before : Nat→ℝ) :
    ∃ t,Kernel.launch (graphAlgebra (RealTarget.model reverseBias upper) c plan) c hasBias useSoftplus
        (.input lower) (.input (RealTarget.upperWord upper)) (entryState 0 before f)=some t ∧
      RealOutputs reverseBias upper before f useSoftplus lower t ∧
      RealFrame reverseBias upper before f t ∧
      Kernel.Frame c c.grid (entryState 0 before f) t ∧ Accesses f ∧ Aligned f := by
  have sizes := sizes_of_contract f.inputs checked
  obtain ⟨t,ran,values,logicalFrame⟩ := RawReal.kernel_correct c sizes plan
    (snapshot 0 before f.inputs) (entryState 0 before f) (entry_graph 0 before f)
    hasBias reverseBias useSoftplus lower upper
  have logical : LogicalOutputs (RealTarget.model reverseBias upper) c plan
      (snapshot 0 before f.inputs) hasBias useSoftplus lower (RealTarget.upperWord upper) t := by
    intro b h k j hb hh hk hj
    simpa only [reference_prepared, reference_cumulative, RealTarget.preparation,
      RealTarget.cumulative_formula] using values b h k j hb hh hk hj
  exact ⟨t,ran,real_outputs reverseBias upper before f useSoftplus lower plan t logical,
    ⟨post_exterior _ before f t,post_input_frame _ before f t⟩,
    logicalFrame,accesses f sizes,aligned f checked⟩

#axiomsClean repaired_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.TensorBridge.Real


namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.TensorBridge.Native
open VeriTile Triton
open Kernel Kernel.Scalar Kernel.Tensor
noncomputable section

/-- Eight fixed-arity scalar families are the numerical boundary. No premise
supplies an output, a successful execution, memory contents or a grid result.
The two Python binary64 values narrow independently at the launcher ABI. -/
theorem repaired_fp32
    (c : Geometry) (f : Buffers c hasBias)
    (checked : InputContract f.inputs.dt.metadata f.inputs.rate.metadata
      (f.inputs.bias.map (·.metadata)) (Int.ofNat c.chunk) integerChunk booleanSoftplus floatLimits)
    (plan : PrefixPlan c.BH c.BC) (reverseBias useSoftplus : Bool)
    (target : Numeric.Target) (unary : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (scalar : Numeric.ScalarLaws target unary ports)
    (lower64 upper64 : BitVec 64) (before : Bytes) :
    let actual := Numeric.nativeModel ports reverseBias
    let reference := Numeric.referenceModel target unary reverseBias
    let lower := ABI.narrow64 lower64
    let upper := ABI.narrow64 upper64
    ∃ t,Kernel.launch (graphAlgebra actual c plan) c hasBias useSoftplus (.input lower) (.input upper)
        (entryState 0 (read32 before) f)=some t ∧
      NativeOutputs reference before f plan useSoftplus lower upper t ∧
      ByteFrame reference before f t ∧
      Kernel.Frame c c.grid (entryState 0 (read32 before) f) t ∧
      bytePost actual before f t=bytePost reference before f t ∧
      bytePost actual before f (entryState 0 (read32 before) f)=before ∧ Accesses f ∧ Aligned f ∧
      ABI.extractBounds (.converted lower64) (.converted upper64)=.ok (lower,upper) := by
  dsimp only
  have sizes := sizes_of_contract f.inputs checked
  have models := Numeric.model_eq target unary ports scalar reverseBias
  obtain ⟨t,ran,values,logicalFrame⟩ := RawNative.kernel_fp32 c sizes plan
    (snapshot 0 (read32 before) f.inputs) (entryState 0 (read32 before) f)
    (entry_graph 0 (read32 before) f) hasBias reverseBias useSoftplus
    target unary ports scalar (ABI.narrow64 lower64) (ABI.narrow64 upper64)
  refine ⟨t,ran,native_outputs _ before f plan useSoftplus _ _ t values,
    byte_frame _ before f t,logicalFrame,?_,byte_entry _ before f,
    accesses f sizes,aligned f checked,rfl⟩
  rw [models]

#axiomsClean repaired_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.TensorBridge.Native
