import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Kernels
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Real

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Correct
open VeriTile VeriTile.Torch VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open RawBoundary HostModel HostStorage
noncomputable section

/-- Checked original-input mathematical wrapper, including all allocation/setup
outcomes, exact returned objects, both independent formulas and caller frame. -/
specification repaired_real
    (dt rate : TensorMetadata) (bias : Option TensorMetadata) (call : Call ℝ)
    (contract : InputContract dt rate bias call.chunk integerChunk booleanSoftplus floatLimits)
    (binding : TensorBinding dt rate bias call) (upper : UpperBound)
    (bound : call.upper=RealProof.upperWord upper)
    (entry : PhysicalMemory ℝ) (old : Nat → Prop)
    (rt : HostModel.Runtime call (logicalInputs call noNegative Neg.neg entry) entry old)
    (real : RealProof.RealRuntime upper rt) (storage : OrdinaryStorage rt)
    (_pointers : PointerBinding dt rate bias storage) :
    ∃ result, guarded dt rate bias call.chunk integerChunk booleanSoftplus floatLimits
      (HostModel.execute call (logicalInputs call noNegative Neg.neg entry) rt)=.ok result ∧
    RealProof.HostSpec call upper (logicalInputs call noNegative Neg.neg entry) rt result ∧
    (∀ g a p (ha : rt.allocate g=.ok a) (hp : rt.inputs g a=p),Facts (storage g a p ha hp)) ∧
    call.dt.shape=dt.shape := by
  refine ⟨_,guarded_eq contract _,RealProof.outcomes call upper bound _ rt real,
    fun g a p ha hp=>actual (storage g a p ha hp)
      (prepared_sizes p (contract.geometrySizes binding g)),binding.dtShape⟩

#axiomsClean repaired_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Correct
