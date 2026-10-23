import bench.repairs.tritonbench_g.lightning_attention.Proofs.Complete
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Contract
namespace VeriTile.Bench.Repairs.LightningAttention.Correct
open VeriTile Triton FP.Structural Host Checked
noncomputable section
abbrev target := @Target.output
abbrev gradients := @Target.loss_coordinate_derivatives

/-- Complete forward Real formula, actual returned Tensor/context and physical
frame. Contiguous Q/K/V use their original descriptors and current cells. -/
specification lightning_forward_real (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (observed : Observations) (admitted : InputContract c.dims observed (forwardTensors a))
    (pointers : PointerBinding observed (forwardTensors a)) (api : ForwardAPI c a env)
    (direct : Direct c a) (before : Nat → Cell ℝ) (fallback : Memory ℝ) :
    Spec.Real (publicForward Real.algebra Real.dotPrimitive c a before fallback=
        some (realForwardResult c a before fallback) ∧
      ForwardPost c a before before (realForwardResult c a before fallback) ∧
      (realForwardResult c a before fallback).output.shape=valueShape c.dims ∧
      (realForwardResult c a before fallback).output.dtype=.fp32 ∧
      (realForwardResult c a before fallback).output.device=c.query.tensor.device ∧
      (realForwardResult c a before fallback).context.query=c.query.tensor ∧
      (realForwardResult c a before fallback).context.key=c.key.tensor ∧
      (realForwardResult c a before fallback).context.value=c.value.tensor) := by
  obtain ⟨he,hp⟩:=forward_endpoint c a env (forward_contract c a env observed admitted pointers api)
    before before fallback (direct_reads Neg.neg c a direct before)
  exact ⟨he,hp,api.output.1,api.output.2.1,api.output.2.2.2.1,direct.query,direct.key,direct.value⟩

/-- Later backward consumes saved Q/K/V and the current materialized cotangent,
then intra followed by inter, returning exactly dQ,dK,dV and true loss derivatives. -/
specification lightning_backward_real (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (observed : Observations) (admitted : InputContract c.dims observed (backwardTensors a))
    (pointers : PointerBinding observed (backwardTensors a)) (api : BackwardAPI c a env)
    (direct : Direct c a) (resolved : c.incoming.lazyNeg=Bool.false)
    (forwardBefore incomingBefore copied : Nat → Cell ℝ) (fallback : Memory ℝ)
    (incoming : IncomingCopy id c a incomingBefore copied)
    (saved : SavedInputs c a forwardBefore copied)
    (copyFrame : CopyFrame (backwardOriginals c a) [a.incomingDense] incomingBefore copied) :
    Spec.Real (∃ result,publicBackward Real.algebra Real.dotPrimitive (savedContext c a) a copied fallback=some result ∧
      BackwardPost c a forwardBefore incomingBefore copied result ∧
      ReturnedVJP c forwardBefore incomingBefore result ∧
      (∀ tensor∈backwardOriginals c a,∀ off,off<span tensor →
        result.physical (tensor.base+off)=incomingBefore (tensor.base+off)) ∧
      (∀ address,¬ChunkLinearAttn.Physical.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address →
        (∀ buf∈backwardBuffers a,¬ChunkLinearAttn.Physical.Contains buf address) →
          result.physical address=incomingBefore address)) := by
  have hi : IncomingCopy Neg.neg c a incomingBefore copied := by
    constructor
    intro b h t f
    simpa [callerRead,resolved] using incoming.incoming b h t f
  exact backward_complete c a env (backward_contract c a env observed admitted pointers api)
    forwardBefore forwardBefore incomingBefore copied fallback (direct_reads Neg.neg c a direct forwardBefore)
    hi saved copyFrame

#specNonCircular target avoiding [publicForward,publicBackward,FP.Structural.exec]
#axiomsClean lightning_forward_real
#axiomsClean lightning_backward_real
end
end VeriTile.Bench.Repairs.LightningAttention.Correct
