import bench.repairs.tritonbench_g.parallel_attention.Proofs.Public
import bench.repairs.tritonbench_g.parallel_attention.Proofs.CallContract
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Physical

namespace VeriTile.Bench.Repairs.ParallelAttention.Checked
noncomputable section
open HostAPI

/-- Only the four profile choices resolved by the bounded SM90 structural
checker; all dimensions, tiles and scaling come from the checked call. -/
structure Syntax where
  postSeed : Bool
  tinyDiagonal : Bool
  scaledSumOnZ : Bool
  paddedCoefficientDirect : Bool

def specialization (c : Call) (d : Domain c) (profile : Syntax) : Public.Specialization :=
  ⟨⟨c.shape,profile.postSeed,profile.tinyDiagonal,profile.scaledSumOnZ,profile.paddedCoefficientDirect⟩,
    d.shape,d.admission⟩

def outputShape (s : Shape) (kind : OutputKind) : List Nat :=
  if kind=.normalizer then [s.B,s.H,s.T] else [s.B,s.H,s.T,kind.width s]

def outputStrides (s : Shape) (kind : OutputKind) : List Int :=
  if kind=.normalizer then [s.H*s.T,s.T,1]
  else [s.H*s.T*kind.width s,s.T*kind.width s,kind.width s,1]

/-- Ordinary successful Torch allocation metadata only. No computed element
value is a field. Input aliases remain allowed, including identical objects. -/
structure Allocated (c : Call) (kind : OutputKind) (base : Nat) (t : TensorView) : Prop where
  supported : TensorKind t
  shape : t.tensor.shape=outputShape c.shape kind
  strides : Machine.Empty c.shape ∨ t.strides=outputStrides c.shape kind
  device : t.tensor.device=c.q.tensor.device
  pointer : t.pointer/4=base
  stored : Stored t (kind.cells c.shape)
  identity : t.identity∉[c.q.identity,c.k.identity,c.v.identity]

structure ForwardTensors where
  output : TensorView
  normalizer : TensorView
  denominator : TensorView
  normalized : TensorView

structure Allocations (c : Call) (out : OutputKind → Nat) (den dst : Nat) (t : ForwardTensors) : Prop where
  output : Allocated c .forward (out .forward) t.output
  normalizer : Allocated c .normalizer (out .normalizer) t.normalizer
  denominator : c.normalize=Bool.true → c.returnBoth=Bool.false → Allocated c .normalizer den t.denominator
  normalized : c.normalize=Bool.true → c.returnBoth=Bool.false → Allocated c .forward dst t.normalized

inductive Returned where
  | both (output normalizer : TensorView)
  | single (output : TensorView)

def returned (c : Call) (t : ForwardTensors) : Returned :=
  if c.returnBoth then .both t.output t.normalizer
  else .single (if c.normalize then t.normalized else t.output)

def Returned.Valid (c : Call) : Returned → Prop
  | .both o z => TensorKind o ∧ TensorKind z ∧
      o.tensor.shape=outputShape c.shape .forward ∧ z.tensor.shape=outputShape c.shape .normalizer ∧
      o.tensor.device=c.q.tensor.device ∧ z.tensor.device=c.q.tensor.device ∧
      o.identity∉[c.q.identity,c.k.identity,c.v.identity] ∧ z.identity∉[c.q.identity,c.k.identity,c.v.identity]
  | .single o => TensorKind o ∧ o.tensor.shape=outputShape c.shape .forward ∧
      o.tensor.device=c.q.tensor.device ∧ o.identity∉[c.q.identity,c.k.identity,c.v.identity]

def Returned.handle (s : Shape) : Returned → HostAPI.Returned
  | .both o z => .both (tensor s .forward (o.pointer/4)) (tensor s .normalizer (z.pointer/4))
  | .single o => .single (tensor s .forward (o.pointer/4))

theorem returned_valid (c : Call) (out : OutputKind → Nat) (den dst : Nat) (t : ForwardTensors)
    (a : Allocations c out den dst t) : (returned c t).Valid c := by
  unfold returned
  split_ifs with both norm
  · exact ⟨a.output.supported,a.normalizer.supported,a.output.shape,a.normalizer.shape,
      a.output.device,a.normalizer.device,a.output.identity,a.normalizer.identity⟩
  · have h:=a.normalized norm (by simpa using both)
    exact ⟨h.supported,h.shape,h.device,h.identity⟩
  · exact ⟨a.output.supported,a.output.shape,a.output.device,a.output.identity⟩

theorem returned_handle (c : Call) (out : OutputKind → Nat) (den dst : Nat) (t : ForwardTensors)
    (a : Allocations c out den dst t) :
    (returned c t).handle c.shape=HostAPI.returned c.shape out dst c.returnBoth c.normalize := by
  unfold returned Returned.handle HostAPI.returned
  split_ifs with both norm <;> dsimp only
  · rw [a.output.pointer,a.normalizer.pointer]
  · rw [(a.normalized norm (by simpa using both)).pointer]
  · rw [a.output.pointer]

structure Result (M : Type) where
  tensors : Returned
  savedIdentities : List Nat
  launchDevice : Nat
  restoredDevice : Nat
  grid : List (Nat×Nat×Nat)
  memory : M

def invoke (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (a : Ops FP32Word) (h : Physical.Heap) : Result Physical.Heap :=
  ⟨returned c t,[c.q.identity,c.k.identity,c.v.identity],c.q.tensor.device.index,c.callerDevice,
    Machine.launches c.shape 128,
    Physical.forward a (specialization c d profile).compiler c.shape (inputs c)
      r.trees r.scratch r.output r.ks r.hs h eps den dst c.returnBoth c.normalize⟩

def logicalForward (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (eps : α) (den dst : Nat) (a : Ops α) (m : Memory α) : HostNative.ForwardResult α :=
  HostNative.forward a (specialization c d profile).compiler c.shape (inputs c)
    r.trees r.scratch r.output r.ks r.hs m eps den dst c.returnBoth c.normalize

theorem invoke_observe (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (a : Ops FP32Word) (h : Physical.Heap) :
    Physical.observe (invoke c d profile r t eps den dst a h).memory=
      (logicalForward c d profile r eps den dst a (Physical.observe h)).api.memory := Physical.observe_forward ..

theorem result_metadata (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : FP32Word) (den dst : Nat) (a : Ops FP32Word) (h : Physical.Heap)
    (alloc : Allocations c r.output den dst t) :
    (invoke c d profile r t eps den dst a h).tensors.Valid c ∧
    (invoke c d profile r t eps den dst a h).savedIdentities=[c.q.identity,c.k.identity,c.v.identity] ∧
    (invoke c d profile r t eps den dst a h).launchDevice=c.q.tensor.device.index ∧
    (invoke c d profile r t eps den dst a h).restoredDevice=c.callerDevice ∧
    (Machine.Empty c.shape → (invoke c d profile r t eps den dst a h).grid=[]) :=
  ⟨returned_valid c r.output den dst t alloc,rfl,rfl,rfl,Machine.empty_no_launch c.shape 128⟩

structure GradientTensors where
  q : TensorView
  k : TensorView
  v : TensorView

structure GradientAllocations (c : Call) (out : OutputKind → Nat) (t : GradientTensors) : Prop where
  q : Allocated c .query (out .query) t.q
  k : Allocated c .key (out .key) t.k
  v : Allocated c .value (out .value) t.v

def GradientTensors.handles (s : Shape) (t : GradientTensors) : Gradients :=
  ⟨tensor s .query (t.q.pointer/4),tensor s .key (t.k.pointer/4),tensor s .value (t.v.pointer/4),none⟩

theorem gradient_handles (c : Call) (out : OutputKind → Nat) (t : GradientTensors)
    (a : GradientAllocations c out t) : t.handles c.shape=gradients c.shape out := by
  unfold GradientTensors.handles gradients
  rw [a.q.pointer,a.k.pointer,a.v.pointer]

theorem gradient_metadata (c : Call) (out : OutputKind → Nat) (t : GradientTensors)
    (a : GradientAllocations c out t) :
    t.q.tensor.shape=[c.shape.B,c.shape.H,c.shape.T,c.shape.K] ∧
    t.k.tensor.shape=[c.shape.B,c.shape.H,c.shape.T,c.shape.K] ∧
    t.v.tensor.shape=[c.shape.B,c.shape.H,c.shape.T,c.shape.V] ∧
    t.q.tensor.device=c.q.tensor.device ∧ t.k.tensor.device=c.q.tensor.device ∧ t.v.tensor.device=c.q.tensor.device ∧
    TensorKind t.q ∧ TensorKind t.k ∧ TensorKind t.v ∧ (t.handles c.shape).scale=none := by
  refine ⟨?_,?_,?_,a.q.device,a.k.device,a.v.device,a.q.supported,a.k.supported,a.v.supported,rfl⟩
  · simpa [outputShape,OutputKind.width] using a.q.shape
  · simpa [outputShape,OutputKind.width] using a.k.shape
  · simpa [outputShape,OutputKind.width] using a.v.shape

/-- Full storage separation supplied by fresh Torch allocations, including
storage gaps outside a caller tensor's logical view. -/
def Avoids (t : TensorView) (word : Nat) : Prop :=
  4*word+4≤t.storageBase ∨ t.storageBase+t.capacity≤4*word

structure CallerStorageFresh (c : Call)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds) (den dst : Nat) : Prop where
  kernel : ∀ t∈[c.q,c.k,c.v],∀ ev∈r.ks,Avoids t (tensorAddress r.scratch ev)
  host : ∀ t∈[c.q,c.k,c.v],∀ ev∈r.hs,Avoids t (tensorAddress r.output ev)
  denominator : c.normalize=Bool.true → c.returnBoth=Bool.false →
    ∀ t∈[c.q,c.k,c.v],∀ i,i<c.shape.B*c.shape.H*c.shape.T → Avoids t (den+i)
  output : c.normalize=Bool.true → c.returnBoth=Bool.false →
    ∀ t∈[c.q,c.k,c.v],∀ i,i<c.shape.B*c.shape.H*c.shape.T*c.shape.V → Avoids t (dst+i)

theorem avoids_word (t : TensorView) (word offset : Nat) (h : Avoids t word) (hi : offset<t.capacity) :
    word≠(t.storageBase+offset)/4 := by unfold Avoids at h;omega

#axiomsClean gradient_handles
#axiomsClean gradient_metadata
#axiomsClean returned_valid
#axiomsClean returned_handle
#axiomsClean invoke_observe
#axiomsClean result_metadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelAttention.Checked
