import bench.repairs.tritonbench_g.lightning_attention.Proofs.EndpointInputs
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def realForwardResult (c : Invocation) (a : Allocations) (copied : Nat → Cell ℝ) (fallback : Memory ℝ) : ForwardResult ℝ :=
  ⟨returnedForward a,savedContext c a,Real.afterForward c.dims (logical a copied fallback),
   publishForward a (Real.afterForward c.dims (logical a copied fallback)) copied⟩
def realBackwardResult (c : Invocation) (a : Allocations) (copied : Nat → Cell ℝ) (fallback : Memory ℝ) : BackwardResult ℝ :=
  ⟨returnedBackward a,Real.backwardMemory c.dims (logical a copied fallback),
   publishBackward a (Real.backwardMemory c.dims (logical a copied fallback)) copied⟩

/-- Independent caller-facing values, exact routing, and physical frame for one forward call. -/
structure ForwardPost (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (result : ForwardResult ℝ) : Prop where
  outputTensor : result.output=a.output
  context : result.context=savedContext c a
  output : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    result.physical (result.output.base+packed c.dims c.dims.E b h t f)=
      .mk .real (Target.output (callerQ c before b h) (callerK c before b h) (callerV c before b h) t f)
  protectedInputs : ∀ tensor∈forwardProtected c a, ∀ o, o<span tensor →
    result.physical (tensor.base+o)=copied (tensor.base+o)
  exterior : ∀ address, ¬ChunkLinearAttn.Physical.Contains (buffer "output" a.output) address →
    result.physical address=copied address

/-- Backward values refer to Q/K/V from the original forward call and G from the
later incoming call. The returned buffers are exactly (dQ,dK,dV). -/
structure BackwardPost (c : Invocation) (a : Allocations) (forwardBefore incomingBefore copied : Nat → Cell ℝ)
    (result : BackwardResult ℝ) : Prop where
  gradients : result.gradients=(a.queryGradient,a.keyGradient,a.valueGradient)
  query : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    result.physical (result.gradients.1.base+packed c.dims c.dims.D b h t f)=
      .mk .real (Target.dq (callerK c forwardBefore b h) (callerV c forwardBefore b h) (callerG c incomingBefore b h) t f)
  key : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    result.physical (result.gradients.2.1.base+packed c.dims c.dims.D b h t f)=
      .mk .real (Target.dk (callerQ c forwardBefore b h) (callerV c forwardBefore b h) (callerG c incomingBefore b h) t f)
  value : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    result.physical (result.gradients.2.2.base+packed c.dims c.dims.E b h t f)=
      .mk .real (Target.dv (callerQ c forwardBefore b h) (callerK c forwardBefore b h) (callerG c incomingBefore b h) t f)
  protectedInputs : ∀ tensor∈backwardProtected c a, ∀ o, o<span tensor →
    result.physical (tensor.base+o)=copied (tensor.base+o)
  exterior : ∀ address, (∀ buf∈backwardBuffers a, ¬ChunkLinearAttn.Physical.Contains buf address) →
    result.physical address=copied address

theorem forward_endpoint (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (hc : ForwardContract c a env) (before copied : Nat → Cell ℝ) (fallback : Memory ℝ)
    (copies : ForwardCopies Neg.neg c a before copied) :
    publicForward Real.algebra Real.dotPrimitive c a copied fallback=some (realForwardResult c a copied fallback) ∧
      ForwardPost c a before copied (realForwardResult c a copied fallback) := by
  refine ⟨publicForward_execution _ _ _ _ _ _,rfl,rfl,?_,?_,?_⟩
  · intro b h t f
    change publishForward a _ copied (a.output.base+packed c.dims c.dims.E b h t f)=_
    rw [forward_published_read c a hc.physical]
    · rw [packed_expand]
      have hp := Real.forward_output c.dims hc.dimensions (logical a copied fallback) (headFin c.dims b h) t f
      rw [query_binding c a before copied fallback copies b h,key_binding c a before copied fallback copies b h,
        value_binding c a before copied fallback copies b h] at hp
      exact hp
    · rw [hc.output.1,packed_expand]
      exact packed_value c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · exact forward_published_inputs c a hc.physical _ copied
  · exact forward_published_frame a _ copied

theorem backward_endpoint (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (hc : BackwardContract c a env) (forwardBefore forwardCopied incomingBefore copied : Nat → Cell ℝ)
    (fallback : Memory ℝ) (copies : ForwardCopies Neg.neg c a forwardBefore forwardCopied)
    (incoming : IncomingCopy Neg.neg c a incomingBefore copied)
    (saved : SavedInputs c a forwardCopied copied) :
    publicBackward Real.algebra Real.dotPrimitive (savedContext c a) a copied fallback=some (realBackwardResult c a copied fallback) ∧
      BackwardPost c a forwardBefore incomingBefore copied (realBackwardResult c a copied fallback) := by
  have bound := saved_copies c a forwardBefore forwardCopied copied copies saved
  refine ⟨publicBackward_execution _ _ _ _ _ _,rfl,?_,?_,?_,?_,?_⟩
  · intro b h t f
    change publishBackward a _ copied (a.queryGradient.base+packed c.dims c.dims.D b h t f)=_
    rw [backward_published_read c a hc.physical _ copied "queryGradient" a.queryGradient (by simp [backwardBuffers])]
    · rw [packed_expand]
      have hp := Real.backward_query c.dims hc.dimensions (logical a copied fallback) (headFin c.dims b h) t f
      rw [key_binding c a forwardBefore copied fallback bound b h,
        value_binding c a forwardBefore copied fallback bound b h,
        incoming_binding c a incomingBefore copied fallback incoming b h] at hp
      exact hp
    · rw [hc.queryGradient.1,packed_expand]
      exact packed_query c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · intro b h t f
    change publishBackward a _ copied (a.keyGradient.base+packed c.dims c.dims.D b h t f)=_
    rw [backward_published_read c a hc.physical _ copied "keyGradient" a.keyGradient (by simp [backwardBuffers])]
    · rw [packed_expand]
      have hp := Real.backward_key c.dims hc.dimensions (logical a copied fallback) (headFin c.dims b h) t f
      rw [query_binding c a forwardBefore copied fallback bound b h,
        value_binding c a forwardBefore copied fallback bound b h,
        incoming_binding c a incomingBefore copied fallback incoming b h] at hp
      exact hp
    · rw [hc.keyGradient.1,packed_expand]
      exact packed_query c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · intro b h t f
    change publishBackward a _ copied (a.valueGradient.base+packed c.dims c.dims.E b h t f)=_
    rw [backward_published_read c a hc.physical _ copied "valueGradient" a.valueGradient (by simp [backwardBuffers])]
    · rw [packed_expand]
      have hp := Real.backward_value c.dims hc.dimensions (logical a copied fallback) (headFin c.dims b h) t f
      rw [query_binding c a forwardBefore copied fallback bound b h,
        key_binding c a forwardBefore copied fallback bound b h,
        incoming_binding c a incomingBefore copied fallback incoming b h] at hp
      exact hp
    · rw [hc.valueGradient.1,packed_expand]
      exact packed_value c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · exact backward_published_inputs c a hc.physical _ copied
  · exact backward_published_frame a _ copied

/-- The values read from the three actually returned physical buffers are the
derivatives of the independently defined scalar loss contracted with later G. -/
theorem returned_loss_derivatives (c : Invocation) (a : Allocations)
    (forwardBefore incomingBefore copied : Nat → Cell ℝ) (result : BackwardResult ℝ)
    (post : BackwardPost c a forwardBefore incomingBefore copied result)
    (b : Fin c.dims.B) (h : Fin c.dims.H) :
    (∀ t f, HasDerivAt (fun ε => Target.loss (Target.bump (callerQ c forwardBefore b h) t f ε)
        (callerK c forwardBefore b h) (callerV c forwardBefore b h) (callerG c incomingBefore b h))
      ((result.physical (result.gradients.1.base+packed c.dims c.dims.D b h t f)).read .real) 0) ∧
    (∀ t f, HasDerivAt (fun ε => Target.loss (callerQ c forwardBefore b h)
        (Target.bump (callerK c forwardBefore b h) t f ε) (callerV c forwardBefore b h) (callerG c incomingBefore b h))
      ((result.physical (result.gradients.2.1.base+packed c.dims c.dims.D b h t f)).read .real) 0) ∧
    (∀ t f, HasDerivAt (fun ε => Target.loss (callerQ c forwardBefore b h) (callerK c forwardBefore b h)
        (Target.bump (callerV c forwardBefore b h) t f ε) (callerG c incomingBefore b h))
      ((result.physical (result.gradients.2.2.base+packed c.dims c.dims.E b h t f)).read .real) 0) := by
  have hd := Target.loss_coordinate_derivatives (callerQ c forwardBefore b h) (callerK c forwardBefore b h)
    (callerV c forwardBefore b h) (callerG c incomingBefore b h)
  refine ⟨?_,?_,?_⟩
  · intro t f; rw [post.query b h t f]; exact hd.1 t f
  · intro t f; rw [post.key b h t f]; exact hd.2.1 t f
  · intro t f; rw [post.value b h t f]; exact hd.2.2 t f
#axiomsClean forward_endpoint
#axiomsClean backward_endpoint
#axiomsClean returned_loss_derivatives
#specNonCircular ForwardPost avoiding [publicForward,modelForward,Real.afterForward]
#specNonCircular BackwardPost avoiding [publicBackward,modelBackward,Real.backwardMemory]
end
end VeriTile.Bench.Repairs.LightningAttention.Host
