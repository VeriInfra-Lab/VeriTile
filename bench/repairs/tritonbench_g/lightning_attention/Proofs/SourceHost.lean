import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativePhases
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Contract
namespace VeriTile.Bench.Repairs.LightningAttention.Checked
open VeriTile Triton FP.Structural Host Native
noncomputable section

def cells (m : WordMemory) : Memory Word := fun r o => .mk .real (m r o)
def logicalWords (a : Allocations) (physical : Nat → Cell Word) (fallback : Memory Word) : WordMemory :=
  fun r off=>(Host.logical a physical fallback r off).read .real

/-- Full typed-source execution, with publication into the actual returned
physical Tensor. No abstract launch-result equality is a premise. -/
def nativeForward (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (current : Nat → Cell Word) (fallback : Memory Word) : Option (ForwardResult Word) := do
  let result ← forwardExecute I plan c.dims (logicalWords a current fallback)
  pure ⟨a.output,savedContext c a,cells result,publishForward a (cells result) current⟩

def nativeBackward (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (current : Nat → Cell Word) (fallback : Memory Word) : Option (BackwardResult Word) := do
  let result ← backwardExecute I plan c.dims (logicalWords a current fallback)
  pure ⟨returnedBackward a,cells result,publishBackward a (cells result) current⟩

theorem native_forward (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (env : LaunchEnvironment) (hc : ForwardContract c a env) (direct : Direct c a)
    (current : Nat → Cell Word) (fallback : Memory Word) :
    ∃ result,nativeForward I plan c a current fallback=some result ∧
      result.output=a.output ∧ result.context.query=c.query.tensor ∧
      result.context.key=c.key.tensor ∧ result.context.value=c.value.tensor ∧
      result.output.shape=valueShape c.dims ∧ result.output.dtype=.fp32 ∧
      result.output.device=c.query.tensor.device ∧
      (∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
        result.physical (a.output.base+packed c.dims c.dims.E b h t f)=
          .mk .real (forwardReference I plan c.dims (logicalWords a current fallback)
            "output" (packed c.dims c.dims.E b h t f))) ∧
      (∀ address,¬ChunkLinearAttn.Physical.Contains (buffer "output" a.output) address →
        result.physical address=current address) := by
  let graph:=forwardReference I plan c.dims (logicalWords a current fallback)
  refine ⟨⟨a.output,savedContext c a,cells graph,publishForward a (cells graph) current⟩,?_,
    rfl,direct.query,direct.key,direct.value,hc.output.1,hc.output.2.1,hc.output.2.2.2.1,?_,?_⟩
  · simp [nativeForward,Native.forward_execution,graph]
  · intro b h t f
    apply forward_published_read c a hc.physical
    have hh : b.val*c.dims.H+h.val<c.dims.B*c.dims.H := by
      have hb:=Nat.mul_le_mul_right c.dims.H (Nat.succ_le_of_lt b.isLt)
      have hbound:=h.isLt
      rw [Nat.succ_mul] at hb
      omega
    have bound:=packed_value c.dims _ _ _ hh t.isLt f.isLt
    rw [hc.output.1]
    convert bound using 1 <;> simp [packed,valueShape] <;> ring
  · exact forward_published_frame a (cells graph) current

theorem native_backward (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (env : LaunchEnvironment) (hc : BackwardContract c a env)
    (current : Nat → Cell Word) (fallback : Memory Word) :
    ∃ result,nativeBackward I plan c a current fallback=some result ∧
      result.gradients=(a.queryGradient,a.keyGradient,a.valueGradient) ∧
      result.gradients.1.shape=queryShape c.dims ∧ result.gradients.2.1.shape=queryShape c.dims ∧
      result.gradients.2.2.shape=valueShape c.dims ∧
      (∀ t∈[result.gradients.1,result.gradients.2.1,result.gradients.2.2],
        t.dtype=.fp32 ∧ t.device=c.query.tensor.device) ∧
      (∀ b∈backwardBuffers a,∀ off,off<b.words →
        result.physical (b.base+off)=.mk .real
          (backwardReference I plan c.dims (logicalWords a current fallback) b.region off)) ∧
      (∀ t∈backwardProtected c a,∀ off,off<span t →
        result.physical (t.base+off)=current (t.base+off)) ∧
      (∀ address,(∀ b∈backwardBuffers a,¬ChunkLinearAttn.Physical.Contains b address) →
        result.physical address=current address) := by
  let graph:=backwardReference I plan c.dims (logicalWords a current fallback)
  refine ⟨⟨returnedBackward a,cells graph,publishBackward a (cells graph) current⟩,?_,rfl,hc.queryGradient.1,hc.keyGradient.1,hc.valueGradient.1,?_,?_,?_,?_⟩
  · simp [nativeBackward,Native.backward_execution,graph]
  · intro t ht
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl|rfl
    · exact ⟨hc.queryGradient.2.1,hc.queryGradient.2.2.2.1⟩
    · exact ⟨hc.keyGradient.2.1,hc.keyGradient.2.2.2.1⟩
    · exact ⟨hc.valueGradient.2.1,hc.valueGradient.2.2.2.1⟩
  · intro b hb off ho
    exact publication_read _ _ hc.physical.publication (cells graph) current b hb off ho
  · exact backward_published_inputs c a hc.physical (cells graph) current
  · exact backward_published_frame a (cells graph) current

#axiomsClean native_forward
#axiomsClean native_backward
end
end VeriTile.Bench.Repairs.LightningAttention.Checked
