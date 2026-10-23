import bench.repairs.tritonbench_g.chunk_linear_attn.Endpoint

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Admission
open VeriTile Triton
open Host
noncomputable section
set_option maxHeartbeats 8000000

structure RuntimeFlags where
  tensorArguments : Bool
  cudaArguments : Bool
  builtinFlags : Bool
  builtinScale : Bool
  finiteScale : Bool
  fp32ScaleRange : Bool

/-- Forward admission and backward cotangent checks, transcribed from the
public wrapper. Incoming cotangents need not have dense strides. -/
def Guards (c : Invocation) (f : RuntimeFlags) : Prop :=
  f.tensorArguments=Bool.true ∧ f.cudaArguments=Bool.true ∧ f.builtinFlags=Bool.true ∧
  f.builtinScale=Bool.true ∧ f.finiteScale=Bool.true ∧ f.fp32ScaleRange=Bool.true ∧
  (0<c.dims.B ∧ 0<c.dims.H ∧ 0<c.dims.T ∧ 16≤c.dims.K ∧ 16≤c.dims.V) ∧
  formedLimit c.dims<2^31 ∧ launchMax c.dims≤65535 ∧
  c.query.shape=queryShape c.dims ∧ c.key.shape=c.query.shape ∧ c.value.shape=valueShape c.dims ∧
  c.query.strides=inputStrides c.dims Bool.false ∧ c.key.strides=c.query.strides ∧
  c.value.strides=inputStrides c.dims Bool.true ∧
  (∀ t∈c.initial,t.shape=stateShape c.dims ∧ t.contiguous=Bool.true) ∧
  (∀ t∈c.incoming,t.shape=valueShape c.dims) ∧
  (∀ t∈c.terminal,t.shape=stateShape c.dims) ∧
  (c.terminal.isSome=Bool.true → c.final=Bool.true) ∧
  (∀ t∈c.forwardInputs++c.incoming.toList++c.terminal.toList,
    t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧ t.device=c.query.device) ∧
  (match c.scale.origin with | .supplied _ => True | .defaultPower K => K=c.dims.K)

instance (c : Invocation) (f : RuntimeFlags) : Decidable (Guards c f) := by
  unfold Guards
  cases c.scale.origin <;> infer_instance

def check (c : Invocation) (f : RuntimeFlags) : Bool := decide (Guards c f)

theorem check_iff (c : Invocation) (f : RuntimeFlags) : check c f=Bool.true ↔ Guards c f := by
  simp [check]

/-- Ordinary tensor storage capacity, supplied by the tensor runtime. Its
content is local readable storage, never an executed source or output value. -/
structure TensorAPI (c : Invocation) : Prop where
  readable : ∀ t∈c.forwardInputs++c.incoming.toList++c.terminal.toList,span t≤t.capacity

theorem contract (c : Invocation) (f : RuntimeFlags) (checked : check c f=Bool.true)
    (api : TensorAPI c) : Host.Contract c := by
  have g:=(check_iff c f).mp checked
  rcases g with ⟨_,_,_,_,_,_,⟨hb,hh,ht,hk,hv⟩,machine,launch,qs,ks,vs,qst,kst,vst,
    initial,incoming,terminal,route,ordinary,origin⟩
  refine ⟨⟨hb,hh,ht,hk,hv⟩,machine,(source_launch_guard c.dims).mp launch,
    qs,ks,vs,qst,kst,vst,initial,incoming,terminal,route,?_,origin⟩
  intro t ht
  exact ⟨(ordinary t ht).1,(ordinary t ht).2.1,(ordinary t ht).2.2,api.readable t ht⟩

/-- The concrete data_ptr observations have been divided by four only after
the runtime alignment guard. This is the ordinary pointer representation. -/
structure PointerObservation (c : Invocation) (a : Allocations) where
  bytes : Physical.Buffer → Nat
  aligned : ∀ b∈outputBuffers a++inputBuffers c a,bytes b=4*b.base

def PointerGuards (c : Invocation) (a : Allocations) (p : PointerObservation c a) : Prop :=
  ∀ b∈outputBuffers a++inputBuffers c a,p.bytes b%4=0 ∧ p.bytes b+4*formedLimit c.dims<2^63

instance (c : Invocation) (a : Allocations) (p : PointerObservation c a) : Decidable (PointerGuards c a p) :=
  inferInstanceAs (Decidable (∀ b∈_,_ ∧ _))

def checkPointers (c : Invocation) (a : Allocations) (p : PointerObservation c a) : Bool :=
  decide (PointerGuards c a p)

theorem pointerABI (c : Invocation) (a : Allocations) (p : PointerObservation c a)
    (checked : checkPointers c a p=Bool.true) : PointerABI c.dims (outputBuffers a++inputBuffers c a) := by
  have g : PointerGuards c a p := by simpa [checkPointers] using checked
  constructor
  intro b hb
  have room:=(g b hb).2
  rw [p.aligned b hb] at room
  omega

/-- Freshness comes from allocation; finite address room is derived from the
actual data_ptr guards rather than being silently assumed by the endpoint. -/
theorem physicalAllocation (c : Invocation) (a : Allocations) (p : PointerObservation c a)
    (checked : checkPointers c a p=Bool.true)
    (separate : ∀ x∈outputBuffers a,∀ y∈outputBuffers a,x≠y → Physical.Separate x y)
    (fresh : ∀ x∈inputBuffers c a,∀ y∈outputBuffers a,Physical.Separate x y) :
    PhysicalAllocation c a := ⟨separate,fresh,pointerABI c a p checked⟩

#axiomsClean contract
#axiomsClean pointerABI
#axiomsClean physicalAllocation
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Admission
