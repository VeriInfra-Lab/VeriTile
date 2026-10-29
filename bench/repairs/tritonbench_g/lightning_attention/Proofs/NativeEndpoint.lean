import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeIngress
import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativePhases
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Complete
namespace VeriTile.Bench.Repairs.LightningAttention.NativeHost
open VeriTile Triton FP.Structural
open Native Host ChunkLinearAttn
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def cells (m : WordMemory) : Memory Word := fun r o => .mk .real (m r o)
def forwardGraph (p : Instructions) (plan : Plan) (d : Dimensions) (m : WordMemory) : WordMemory :=
  gridReference p plan .forward d (grid .forward d) m
def backwardGraph (p : Instructions) (plan : Plan) (d : Dimensions) (m : WordMemory) : WordMemory :=
  gridReference p plan .inter d (grid .inter d) (gridReference p plan .intra d (grid .intra d) m)

def forwardCall (runtime : ForwardRuntime) (c : Invocation) (a : Allocations)
    (copied : Nat → Cell Word) (fallback : Memory Word) : Option (ForwardResult Word) := do
  let result ← runtime.launch (logicalWords a copied fallback)
  pure ⟨returnedForward a,savedContext c a,cells result,publishForward a (cells result) copied⟩
def backwardCall (runtime : BackwardRuntime) (ctx : Context) (a : Allocations)
    (copied : Nat → Cell Word) (fallback : Memory Word) : Option (BackwardResult Word) := do
  let alloc := withContext ctx a
  let intra ← runtime.intra (logicalWords alloc copied fallback)
  let result ← runtime.inter intra
  pure ⟨returnedBackward alloc,cells result,publishBackward alloc (cells result) copied⟩

def forwardResult (p : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (copied : Nat → Cell Word) (fallback : Memory Word) : ForwardResult Word :=
  let result := cells (forwardGraph p plan c.dims (logicalWords a copied fallback))
  ⟨returnedForward a,savedContext c a,result,publishForward a result copied⟩
def backwardResult (p : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (copied : Nat → Cell Word) (fallback : Memory Word) : BackwardResult Word :=
  let result := cells (backwardGraph p plan c.dims (logicalWords a copied fallback))
  ⟨returnedBackward a,result,publishBackward a result copied⟩

/-- Native returned tensor cells equal the explicit selected fixed-ISA graph.
Input-copy and original physical frame claims are included independently. -/
structure ForwardPost (c : Invocation) (a : Allocations) (before copied : Nat → Cell Word)
    (graph : WordMemory) (result : ForwardResult Word) : Prop where
  outputTensor : result.output=a.output
  context : result.context=savedContext c a
  copies : ForwardCopies negateWord c a before copied
  output : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    result.physical (result.output.base+packed c.dims c.dims.E b h t f)=
      .mk .real (graph "output" (packed c.dims c.dims.E b h t f))
  originalInputs : ∀ tensor∈forwardOriginals c, ∀ o, o<span tensor →
    result.physical (tensor.base+o)=before (tensor.base+o)
  exterior : ∀ address, (∀ tensor∈forwardDestinations a,
      ¬Physical.Contains ⟨"copy",tensor.base,span tensor⟩ address) →
    ¬Physical.Contains (buffer "output" a.output) address → result.physical address=before address

structure BackwardPost (c : Invocation) (a : Allocations) (forwardBefore incomingBefore copied : Nat → Cell Word)
    (graph : WordMemory) (result : BackwardResult Word) : Prop where
  gradients : result.gradients=(a.queryGradient,a.keyGradient,a.valueGradient)
  copies : ForwardCopies negateWord c a forwardBefore copied
  incoming : IncomingCopy negateWord c a incomingBefore copied
  query : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    result.physical (result.gradients.1.base+packed c.dims c.dims.D b h t f)=
      .mk .real (graph "queryGradient" (packed c.dims c.dims.D b h t f))
  key : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    result.physical (result.gradients.2.1.base+packed c.dims c.dims.D b h t f)=
      .mk .real (graph "keyGradient" (packed c.dims c.dims.D b h t f))
  value : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    result.physical (result.gradients.2.2.base+packed c.dims c.dims.E b h t f)=
      .mk .real (graph "valueGradient" (packed c.dims c.dims.E b h t f))
  originalInputs : ∀ tensor∈backwardOriginals c a, ∀ o, o<span tensor →
    result.physical (tensor.base+o)=incomingBefore (tensor.base+o)
  exterior : ∀ address, ¬Physical.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address →
    (∀ buf∈backwardBuffers a, ¬Physical.Contains buf address) → result.physical address=incomingBefore address

theorem forward_complete (actual selected : Instructions) (plan : Plan) (runtime : ForwardRuntime)
    (negate : Word → Word) (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (hc : ForwardContract c a env) (binding : ForwardBinding actual plan c.dims runtime)
    (numeric : LocalConformance actual selected) (negative : NegationConformance negate)
    (before copied : Nat → Cell Word) (fallback : Memory Word)
    (copies : ForwardCopies negate c a before copied)
    (copyFrame : CopyFrame (forwardOriginals c) (forwardDestinations a) before copied) :
    ∃ result, forwardCall runtime c a copied fallback=some result ∧
      ForwardPost c a before copied (forwardGraph selected plan c.dims (logicalWords a copied fallback)) result := by
  refine ⟨forwardResult selected plan c a copied fallback,?_,rfl,rfl,forward_copies negate negative c a before copied copies,?_,?_,?_⟩
  · simp only [forwardCall,binding.launch,bind,Option.bind_some,pure]
    rw [instructions_eq actual selected numeric]
    rfl
  · intro b h t f
    apply forward_published_read c a hc.physical
    rw [hc.output.1,packed_expand]
    exact packed_value c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · intro tensor ht o ho
    refine (forward_published_inputs c a hc.physical _ copied tensor ?_ o ho).trans (copyFrame.inputs tensor ht o ho)
    simp only [forwardOriginals,List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl|rfl <;> simp [forwardProtected]
  · intro address hcopies hout
    exact (forward_published_frame a _ copied address hout).trans (copyFrame.exterior address hcopies)

theorem backward_complete (actual selected : Instructions) (plan : Plan) (runtime : BackwardRuntime)
    (negate : Word → Word) (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (hc : BackwardContract c a env) (binding : BackwardBinding actual plan c.dims runtime)
    (numeric : LocalConformance actual selected) (negative : NegationConformance negate)
    (forwardBefore forwardCopied incomingBefore copied : Nat → Cell Word) (fallback : Memory Word)
    (copies : ForwardCopies negate c a forwardBefore forwardCopied)
    (incoming : IncomingCopy negate c a incomingBefore copied)
    (saved : SavedInputs c a forwardCopied copied)
    (copyFrame : CopyFrame (backwardOriginals c a) [a.incomingDense] incomingBefore copied) :
    ∃ result, backwardCall runtime (savedContext c a) a copied fallback=some result ∧
      BackwardPost c a forwardBefore incomingBefore copied
        (backwardGraph selected plan c.dims (logicalWords a copied fallback)) result := by
  have hcps : ForwardCopies negate c a forwardBefore copied :=
    ⟨fun b h t f => (saved.query b h t f).trans (copies.query b h t f),
     fun b h t f => (saved.key b h t f).trans (copies.key b h t f),
     fun b h t f => (saved.value b h t f).trans (copies.value b h t f)⟩
  refine ⟨backwardResult selected plan c a copied fallback,?_,rfl,
    forward_copies negate negative c a forwardBefore copied hcps,
    incoming_copy negate negative c a incomingBefore copied incoming,?_,?_,?_,?_,?_⟩
  · unfold backwardCall
    rw [with_saved]
    simp only [binding.intra,binding.inter,bind,Option.bind_some,pure]
    rw [instructions_eq actual selected numeric]
    rfl
  · intro b h t f
    apply backward_published_read c a hc.physical _ copied "queryGradient" a.queryGradient (by simp [backwardBuffers])
    rw [hc.queryGradient.1,packed_expand]
    exact packed_query c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · intro b h t f
    apply backward_published_read c a hc.physical _ copied "keyGradient" a.keyGradient (by simp [backwardBuffers])
    rw [hc.keyGradient.1,packed_expand]
    exact packed_query c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · intro b h t f
    apply backward_published_read c a hc.physical _ copied "valueGradient" a.valueGradient (by simp [backwardBuffers])
    rw [hc.valueGradient.1,packed_expand]
    exact packed_value c.dims _ _ _ (headFin c.dims b h).isLt t.isLt f.isLt
  · intro tensor ht o ho
    refine (backward_published_inputs c a hc.physical _ copied tensor ?_ o ho).trans (copyFrame.inputs tensor ht o ho)
    simp only [backwardOriginals,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with ht|(rfl|rfl|rfl|rfl)
    · simp [backwardProtected,protectedTensors,ht]
    all_goals simp [backwardProtected,protectedTensors,Allocations.inputs]
  · intro address hcopy hout
    refine (backward_published_frame a _ copied address hout).trans (copyFrame.exterior address ?_)
    intro tensor ht
    simpa only [List.mem_singleton.mp ht] using hcopy
#axiomsClean forward_complete
#axiomsClean backward_complete
#specNonCircular ForwardPost avoiding [forwardCall,forwardResult]
#specNonCircular BackwardPost avoiding [backwardCall,backwardResult]
end
end VeriTile.Bench.Repairs.LightningAttention.NativeHost
