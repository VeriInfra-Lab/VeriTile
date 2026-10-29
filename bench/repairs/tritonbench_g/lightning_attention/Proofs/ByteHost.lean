import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeBytes
namespace VeriTile.Bench.Repairs.LightningAttention.ByteHost
open VeriTile Triton FP.Structural Host Native Checked NativeBytes
open VeriTile.Bench.Repairs.ChunkLinearAttn
noncomputable section

structure ForwardResult where
  words : Host.ForwardResult Word
  bytes : Bytes
structure BackwardResult where
  words : Host.BackwardResult Word
  bytes : Bytes

def forward (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (current : Bytes) (fallback : Memory Word) : Option ForwardResult := do
  let result ← nativeForward I plan c a (fromBytes current) fallback
  pure ⟨result,NativeBytes.publish (forwardBuffers a) (fun r off=>(result.memory r off).read .real) current⟩

def backward (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (current : Bytes) (fallback : Memory Word) : Option BackwardResult := do
  let result ← nativeBackward I plan c a (fromBytes current) fallback
  pure ⟨result,NativeBytes.publish (backwardBuffers a) (fun r off=>(result.memory r off).read .real) current⟩

theorem forward_lift (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (current : Bytes) (fallback : Memory Word) (result : Host.ForwardResult Word)
    (execution : nativeForward I plan c a (fromBytes current) fallback=some result) :
    ∃ bytes,forward I plan c a current fallback=some ⟨result,bytes⟩ ∧
      fromBytes bytes=result.physical ∧
      (∀ address,¬NativeBytes.Contains (buffer "output" a.output) address → bytes address=current address) := by
  have model := execution
  simp only [nativeForward,Native.forward_execution,bind,Option.bind_some,pure,Option.some.injEq] at model
  subst result
  refine ⟨NativeBytes.publish (forwardBuffers a)
    (forwardReference I plan c.dims (logicalWords a (fromBytes current) fallback)) current,?_,?_,?_⟩
  · simp [forward,nativeForward,Native.forward_execution,Checked.cells,Cell.read]
  · simpa [Checked.cells,Cell.read,publishForward,publishBackward] using
      load_publish (forwardBuffers a) (forwardReference I plan c.dims (logicalWords a (fromBytes current) fallback)) current
  · intro address outside
    apply NativeBytes.publish_frame
    simpa only [forwardBuffers,List.mem_singleton] using (fun b (hb : b=buffer "output" a.output)=>hb ▸ outside)

theorem backward_lift (I : Instructions) (plan : Plan) (c : Invocation) (a : Allocations)
    (current : Bytes) (fallback : Memory Word) (result : Host.BackwardResult Word)
    (execution : nativeBackward I plan c a (fromBytes current) fallback=some result) :
    ∃ bytes,backward I plan c a current fallback=some ⟨result,bytes⟩ ∧
      fromBytes bytes=result.physical ∧
      (∀ address,(∀ b∈backwardBuffers a,¬NativeBytes.Contains b address) → bytes address=current address) := by
  have model := execution
  simp only [nativeBackward,Native.backward_execution,bind,Option.bind_some,pure,Option.some.injEq] at model
  subst result
  refine ⟨NativeBytes.publish (backwardBuffers a)
    (backwardReference I plan c.dims (logicalWords a (fromBytes current) fallback)) current,?_,?_,?_⟩
  · simp [backward,nativeBackward,Native.backward_execution,Checked.cells,Cell.read]
  · simpa [Checked.cells,Cell.read,publishForward,publishBackward] using
      load_publish (backwardBuffers a) (backwardReference I plan c.dims (logicalWords a (fromBytes current) fallback)) current
  · exact NativeBytes.publish_frame _ _ current

theorem byte_of_word (bytes : Bytes) (physical : Nat → Cell Word) (binding : fromBytes bytes=physical)
    (a : Nat) (value : Word) (h : physical a=.mk .real value) (i : Fin 4) :
    bytes (4*a+i.val)=encode value i := by
  have cell := (congrFun binding a).trans h
  have eq : NativeBytes.load bytes a=value := by
    simpa [fromBytes,Cell.read] using congrArg (fun v : Cell Word=>v.read .real) cell
  rw [←original_bytes bytes a i,eq]

theorem byte_frame_of_cells (a b : Bytes) (wordAddress : Nat)
    (eq : fromBytes a wordAddress=fromBytes b wordAddress) (i : Fin 4) :
    a (4*wordAddress+i.val)=b (4*wordAddress+i.val) := by
  have h := congrArg (fun cell=>cell.read .real) eq
  simp [fromBytes,Cell.read] at h
  rw [←original_bytes a wordAddress i,←original_bytes b wordAddress i,h]

/-- A byte-level local contract for the retained cotangent contiguous copy.
It describes only input coordinates and the copy footprint, never a kernel output. -/
structure IncomingBytes (c : Invocation) (a : Allocations) (before copied : Bytes) : Prop where
  values : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E) (i : Fin 4),
    copied (4*(a.incomingDense.base+packed c.dims c.dims.E b h t f)+i.val)=
      before (4*(c.incoming.tensor.base+offset4 c.incoming.tensor b h t f)+i.val)
  inputFrame : ∀ tensor∈backwardProtected c a,∀ off,off<span tensor → ∀ i : Fin 4,
    copied (4*(tensor.base+off)+i.val)=before (4*(tensor.base+off)+i.val)
  frame : ∀ byte,¬NativeBytes.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ byte →
    copied byte=before byte

theorem incoming_words (c : Invocation) (a : Allocations) (before copied : Bytes)
    (copy : IncomingBytes c a before copied) (resolved : c.incoming.lazyNeg=Bool.false) :
    IncomingCopy id c a (fromBytes before) (fromBytes copied) := by
  constructor
  intro b h t f
  simp only [callerRead,resolved,Bool.false_eq_true,↓reduceIte]
  simp only [fromBytes,Cell.read,↓reduceDIte]
  unfold NativeBytes.load
  apply congrArg decode
  funext i
  exact copy.values b h t f i

theorem copy_inputs_words (c : Invocation) (a : Allocations) (before copied : Bytes)
    (copy : IncomingBytes c a before copied) : ∀ tensor∈backwardProtected c a,∀ off,off<span tensor →
    fromBytes copied (tensor.base+off)=fromBytes before (tensor.base+off) := by
  intro tensor ht off ho
  unfold fromBytes NativeBytes.load
  congr 1
  apply congrArg decode
  funext i
  exact copy.inputFrame tensor ht off ho i

theorem copy_frame_words (c : Invocation) (a : Allocations) (before copied : Bytes)
    (copy : IncomingBytes c a before copied) (wordAddress : Nat)
    (outside : ¬Physical.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ wordAddress) :
    fromBytes copied wordAddress=fromBytes before wordAddress := by
  unfold fromBytes NativeBytes.load
  congr 1
  apply congrArg decode
  funext i
  apply copy.frame
  rw [NativeBytes.contains_iff]
  simpa only [show (4*wordAddress+i.val)/4=wordAddress by omega] using outside

#axiomsClean forward_lift
#axiomsClean backward_lift
#axiomsClean byte_of_word
#axiomsClean byte_frame_of_cells
#axiomsClean incoming_words
#axiomsClean copy_inputs_words
#axiomsClean copy_frame_words
#specNonCircular IncomingBytes avoiding [forward,backward,nativeForward,nativeBackward]
end
end VeriTile.Bench.Repairs.LightningAttention.ByteHost
