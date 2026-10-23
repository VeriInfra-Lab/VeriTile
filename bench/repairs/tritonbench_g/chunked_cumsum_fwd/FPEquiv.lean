import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Kernels
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.FP

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.FPEquiv
open VeriTile VeriTile.Torch VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open RawBoundary HostModel HostStorage
noncomputable section

def negate (w : Word) : Word := w ^^^ BitVec.ofNat 32 (2^31)

/-- Complete FP32 source and flat-byte result. The runtime clamp ABI and actual
ordered scan/FMA plans are retained. Scalar numerical conformance is local. -/
specification repaired_fp32
    (dt rate : TensorMetadata) (bias : Option TensorMetadata) (raw : Call RawABI.Scalar)
    (contract : InputContract dt rate bias raw.chunk integerChunk booleanSoftplus floatLimits)
    (binding : TensorBinding dt rate bias (RawABI.materialize raw))
    (entry : PhysicalMemory Word) (old : Nat → Prop)
    (rt : HostModel.Runtime (RawABI.materialize raw)
      (logicalInputs (RawABI.materialize raw) noNegative negate entry) entry old)
    (storage : OrdinaryStorage rt) (_pointers : PointerBinding dt rate bias storage) :
    let logical := logicalInputs (RawABI.materialize raw) noNegative negate entry
    let native := FPProof.guarded raw logical rt
    ∃ result, guarded dt rate bias raw.chunk integerChunk booleanSoftplus floatLimits
      (HostModel.execute (RawABI.materialize raw) logical native)=.ok result ∧
      FPProof.HostSpec (RawABI.materialize raw) logical native storage result ∧
      (∀ g a p plans,native.setup g a p=.ok plans → RawABI.extractBounds raw.lower raw.upper=
        .ok ((RawABI.materialize raw).lower,(RawABI.materialize raw).upper)) ∧
      (RawABI.materialize raw).dt.shape=dt.shape := by
  exact ⟨_,guarded_eq contract _,FPProof.outcomes _ _ (FPProof.guarded raw _ rt) storage
      (InputContract.geometrySizes (call:=RawABI.materialize raw) contract binding),
    fun g a p plans ok=>FPProof.bounds_binding raw _ rt g a p plans ok,binding.dtShape⟩

#axiomsClean repaired_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.FPEquiv
