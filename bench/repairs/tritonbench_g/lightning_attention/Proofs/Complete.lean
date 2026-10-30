import bench.repairs.tritonbench_g.lightning_attention.Proofs.PublicFrame
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural ChunkLinearAttn
noncomputable section
set_option maxHeartbeats 6000000

def ReturnedVJP (c : Invocation) (forwardBefore incomingBefore : Nat → Cell ℝ)
    (result : BackwardResult ℝ) : Prop :=
  ∀ (b : Fin c.dims.B) (h : Fin c.dims.H),
    (∀ t f, HasDerivAt (fun ε => Target.loss (Target.bump (callerQ c forwardBefore b h) t f ε)
        (callerK c forwardBefore b h) (callerV c forwardBefore b h) (callerG c incomingBefore b h))
      ((result.physical (result.gradients.1.base+packed c.dims c.dims.D b h t f)).read .real) 0) ∧
    (∀ t f, HasDerivAt (fun ε => Target.loss (callerQ c forwardBefore b h)
        (Target.bump (callerK c forwardBefore b h) t f ε) (callerV c forwardBefore b h) (callerG c incomingBefore b h))
      ((result.physical (result.gradients.2.1.base+packed c.dims c.dims.D b h t f)).read .real) 0) ∧
    (∀ t f, HasDerivAt (fun ε => Target.loss (callerQ c forwardBefore b h) (callerK c forwardBefore b h)
        (Target.bump (callerV c forwardBefore b h) t f ε) (callerG c incomingBefore b h))
      ((result.physical (result.gradients.2.2.base+packed c.dims c.dims.E b h t f)).read .real) 0)

theorem forward_complete (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (hc : ForwardContract c a env) (before copied : Nat → Cell ℝ) (fallback : Memory ℝ)
    (copies : ForwardCopies Neg.neg c a before copied)
    (copyFrame : CopyFrame (forwardOriginals c) (forwardDestinations a) before copied) :
    ∃ result, publicForward Real.algebra Real.dotPrimitive c a copied fallback=some result ∧
      ForwardPost c a before copied result ∧
      (∀ tensor∈forwardOriginals c, ∀ o, o<span tensor → result.physical (tensor.base+o)=before (tensor.base+o)) ∧
      (∀ address, (∀ tensor∈forwardDestinations a, ¬Physical.Contains ⟨"copy",tensor.base,span tensor⟩ address) →
        ¬Physical.Contains (buffer "output" a.output) address → result.physical address=before address) := by
  obtain ⟨he,hp⟩ := forward_endpoint c a env hc before copied fallback copies
  exact ⟨realForwardResult c a copied fallback,he,hp,
    forward_original_inputs c a before copied _ hp copyFrame,
    forward_complete_frame c a before copied _ hp copyFrame⟩

theorem backward_complete (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (hc : BackwardContract c a env) (forwardBefore forwardCopied incomingBefore copied : Nat → Cell ℝ)
    (fallback : Memory ℝ) (copies : ForwardCopies Neg.neg c a forwardBefore forwardCopied)
    (incoming : IncomingCopy Neg.neg c a incomingBefore copied)
    (saved : SavedInputs c a forwardCopied copied)
    (copyFrame : CopyFrame (backwardOriginals c a) [a.incomingDense] incomingBefore copied) :
    ∃ result, publicBackward Real.algebra Real.dotPrimitive (savedContext c a) a copied fallback=some result ∧
      BackwardPost c a forwardBefore incomingBefore copied result ∧
      ReturnedVJP c forwardBefore incomingBefore result ∧
      (∀ tensor∈backwardOriginals c a, ∀ o, o<span tensor → result.physical (tensor.base+o)=incomingBefore (tensor.base+o)) ∧
      (∀ address, ¬Physical.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address →
        (∀ buf∈backwardBuffers a, ¬Physical.Contains buf address) → result.physical address=incomingBefore address) := by
  obtain ⟨he,hp⟩ := backward_endpoint c a env hc forwardBefore forwardCopied incomingBefore copied fallback copies incoming saved
  exact ⟨realBackwardResult c a copied fallback,he,hp,
    returned_loss_derivatives c a forwardBefore incomingBefore copied _ hp,
    backward_original_inputs c a forwardBefore incomingBefore copied _ hp copyFrame,
    backward_complete_frame c a forwardBefore incomingBefore copied _ hp copyFrame⟩
#axiomsClean forward_complete
#axiomsClean backward_complete
#specNonCircular ReturnedVJP avoiding [publicBackward,modelBackward,Real.backwardMemory,Target.dq,Target.dk,Target.dv]
end
end VeriTile.Bench.Repairs.LightningAttention.Host
