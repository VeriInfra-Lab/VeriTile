import bench.repairs.tritonbench_g.lightning_attention.Proofs.Endpoint
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural
open ChunkLinearAttn
noncomputable section
set_option maxHeartbeats 6000000
/-- The ordinary library-copy frame covers allocation and materialization,
which precede each Triton launch. It refers only to spans and input memory. -/
structure CopyFrame {α : Type} (protectedTensors destinations : List Tensor)
    (before copied : Nat → Cell α) : Prop where
  inputs : ∀ tensor∈protectedTensors, ∀ o, o<span tensor → copied (tensor.base+o)=before (tensor.base+o)
  exterior : ∀ address, (∀ tensor∈destinations,
    ¬Physical.Contains ⟨"copy",tensor.base,span tensor⟩ address) → copied address=before address

def forwardOriginals (c : Invocation) : List Tensor := [c.query.tensor,c.key.tensor,c.value.tensor]
def forwardDestinations (a : Allocations) : List Tensor := [a.queryDense,a.keyDense,a.valueDense]
def backwardOriginals (c : Invocation) (a : Allocations) : List Tensor :=
  c.inputs++[a.queryDense,a.keyDense,a.valueDense,a.output]

theorem forward_original_inputs (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (result : ForwardResult ℝ) (post : ForwardPost c a before copied result)
    (copyFrame : CopyFrame (forwardOriginals c) (forwardDestinations a) before copied)
    (tensor : Tensor) (ht : tensor∈forwardOriginals c) (o : Nat) (ho : o<span tensor) :
    result.physical (tensor.base+o)=before (tensor.base+o) := by
  refine (post.protectedInputs tensor ?_ o ho).trans (copyFrame.inputs tensor ht o ho)
  simp only [forwardOriginals,List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with rfl|rfl|rfl <;> simp [forwardProtected]

theorem forward_complete_frame (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (result : ForwardResult ℝ) (post : ForwardPost c a before copied result)
    (copyFrame : CopyFrame (forwardOriginals c) (forwardDestinations a) before copied)
    (address : Nat) (hc : ∀ tensor∈forwardDestinations a,
      ¬Physical.Contains ⟨"copy",tensor.base,span tensor⟩ address)
    (ho : ¬Physical.Contains (buffer "output" a.output) address) : result.physical address=before address :=
  (post.exterior address ho).trans (copyFrame.exterior address hc)

theorem backward_original_inputs (c : Invocation) (a : Allocations)
    (forwardBefore incomingBefore copied : Nat → Cell ℝ) (result : BackwardResult ℝ)
    (post : BackwardPost c a forwardBefore incomingBefore copied result)
    (copyFrame : CopyFrame (backwardOriginals c a) [a.incomingDense] incomingBefore copied)
    (tensor : Tensor) (ht : tensor∈backwardOriginals c a) (o : Nat) (ho : o<span tensor) :
    result.physical (tensor.base+o)=incomingBefore (tensor.base+o) :=
by
  refine (post.protectedInputs tensor ?_ o ho).trans (copyFrame.inputs tensor ht o ho)
  simp only [backwardOriginals,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with ht|(rfl|rfl|rfl|rfl)
  · simp [backwardProtected,protectedTensors,ht]
  all_goals simp [backwardProtected,protectedTensors,Allocations.inputs]

theorem backward_complete_frame (c : Invocation) (a : Allocations)
    (forwardBefore incomingBefore copied : Nat → Cell ℝ) (result : BackwardResult ℝ)
    (post : BackwardPost c a forwardBefore incomingBefore copied result)
    (copyFrame : CopyFrame (backwardOriginals c a) [a.incomingDense] incomingBefore copied)
    (address : Nat) (hc : ¬Physical.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address)
    (ho : ∀ buf∈backwardBuffers a, ¬Physical.Contains buf address) :
    result.physical address=incomingBefore address := by
  refine (post.exterior address ho).trans (copyFrame.exterior address ?_)
  intro tensor ht
  simpa only [List.mem_singleton.mp ht] using hc
#axiomsClean forward_original_inputs
#axiomsClean forward_complete_frame
#axiomsClean backward_original_inputs
#axiomsClean backward_complete_frame
#specNonCircular CopyFrame avoiding [publicForward,publicBackward,publishForward,publishBackward]
end
end VeriTile.Bench.Repairs.LightningAttention.Host
