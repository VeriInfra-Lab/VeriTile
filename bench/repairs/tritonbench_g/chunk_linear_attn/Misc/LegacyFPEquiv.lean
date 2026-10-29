import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.Admission

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open Host
noncomputable section

/-- The full FP32 forward/backward graph with actual scale bits, local scalar
ISA leaves, exact returned allocation identities, input preservation and frame. -/
theorem wrapper_fp32 (I : ISA (BitVec 32)) (profile : NativeTerms.FP32Profile)
    (numeric : NativeTerms.PrimitiveLeaves I profile) (c : Invocation)
    (flags : Admission.RuntimeFlags) (checked : Admission.check c flags=Bool.true)
    (api : Admission.TensorAPI c) (a : Allocations) (ac : AllocationContract c a)
    (p : Admission.PointerObservation c a) (pointers : Admission.checkPointers c a p=Bool.true)
    (separate : ∀ x∈outputBuffers a,∀ y∈outputBuffers a,x≠y → Physical.Separate x y)
    (fresh : ∀ x∈inputBuffers c a,∀ y∈outputBuffers a,Physical.Separate x y)
    (before : Nat → BitVec 32) (fallback : RegionName → Nat → BitVec 32) :
    NativePost profile c a before fallback (nativePublished I c a before fallback) :=
  native_complete I profile numeric c (Admission.contract c flags checked api) a ac
    (Admission.physicalAllocation c a p pointers separate fresh) before fallback

#axiomsClean wrapper_fp32
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.FPEquiv
