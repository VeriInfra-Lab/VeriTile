import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostAPI
import bench.repairs.tritonbench_g.lightning_attention.Proofs.DirectInputs
namespace VeriTile.Bench.Repairs.LightningAttention.Checked
open VeriTile Triton FP.Structural Host
noncomputable section

/-- Classified Python observations. Forward inputs are contiguous. Backward
retains the original cotangent materialization before checking launch pointers. -/
structure Observations where
  ordinaryCUDAFP32 : Bool
  sameDeviceShapes : Bool
  contiguousForward : Bool
  resolvedViews : Bool
  pointer : Tensor → Nat

def InputContract (d : Dimensions) (o : Observations) (tensors : List Tensor) : Prop :=
  o.ordinaryCUDAFP32=Bool.true ∧ o.sameDeviceShapes=Bool.true ∧
  o.contiguousForward=Bool.true ∧ o.resolvedViews=Bool.true ∧
  0<d.B ∧ 0<d.H ∧ 0<d.N ∧ 16≤d.D ∧ 16≤d.E ∧
  (∃ k,d.D=2^k) ∧ (∃ k,d.E=2^k) ∧ d.NT≤65535 ∧ d.NV≤65535 ∧
  d.formedLimit<2^31 ∧ (∀ t∈tensors,o.pointer t%4=0 ∧ o.pointer t+4*d.formedLimit<2^63)

def check (d : Dimensions) (o : Observations) (tensors : List Tensor) : Bool := by
  classical
  exact decide (InputContract d o tensors)

theorem check_iff (d : Dimensions) (o : Observations) (tensors : List Tensor) :
    check d o tensors=Bool.true ↔ InputContract d o tensors := by
  classical
  simp [check]

theorem dimensions (d : Dimensions) (o : Observations) (ts : List Tensor)
    (h : InputContract d o ts) : Host.Contract d := by
  rcases h with ⟨_,_,_,_,hb,hh,hn,hd,he,dp,ep,nt,nv,hm,_⟩
  exact ⟨hb,hh,hn,hd,he,dp,ep,nt,nv,hm⟩

def forwardTensors (a : Allocations) : List Tensor := [a.queryDense,a.keyDense,a.valueDense,a.output]
def backwardTensors (a : Allocations) : List Tensor :=
  [a.queryDense,a.keyDense,a.valueDense,a.incomingDense,a.queryGradient,a.keyGradient,a.valueGradient]

def PointerBinding (o : Observations) (ts : List Tensor) : Prop := ∀ t∈ts,o.pointer t=4*t.base

theorem pointer_room (d : Dimensions) (o : Observations) (ts : List Tensor)
    (h : InputContract d o ts) (binding : PointerBinding o ts) (t : Tensor) (ht : t∈ts) :
    t.base+d.formedLimit<2^61 := by
  rcases h with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,hp⟩
  have hb:=(hp t ht).2
  rw [binding t ht] at hb
  norm_num at hb ⊢
  omega

theorem aligned (d : Dimensions) (o : Observations) (ts : List Tensor)
    (h : InputContract d o ts) (t : Tensor) (ht : t∈ts) (off : Nat) :
    (o.pointer t+4*off)%4=0 := by
  rcases h with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,hp⟩
  have ha:=(hp t ht).1
  omega

theorem forward_pointers (c : Invocation) (a : Allocations) (o : Observations)
    (h : InputContract c.dims o (forwardTensors a)) (binding : PointerBinding o (forwardTensors a)) :
    PointerABI c.dims (forwardDeviceBuffers a) := by
  constructor
  intro b hb
  simp only [forwardDeviceBuffers,forwardBuffers,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hb
  rcases hb with (rfl|rfl|rfl)|rfl <;>
    exact pointer_room c.dims o _ h binding _ (by simp [forwardTensors])

theorem backward_pointers (c : Invocation) (a : Allocations) (o : Observations)
    (h : InputContract c.dims o (backwardTensors a)) (binding : PointerBinding o (backwardTensors a)) :
    PointerABI c.dims (backwardDeviceBuffers a) := by
  constructor
  intro b hb
  simp only [backwardDeviceBuffers,backwardBuffers,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hb
  rcases hb with (rfl|rfl|rfl|rfl)|(rfl|rfl|rfl) <;>
    exact pointer_room c.dims o _ h binding _ (by simp [backwardTensors])

/-- Ordinary Tensor/allocator metadata and the selected-device runtime state.
These records carry no values, execution equality, or calculated result. -/
structure ForwardAPI (c : Invocation) (a : Allocations) (env : LaunchEnvironment) : Prop where
  launch : LaunchContract c.query.tensor.device env
  query : ValidView c.query.tensor.device (queryShape c.dims) c.query
  key : ValidView c.query.tensor.device (queryShape c.dims) c.key
  value : ValidView c.query.tensor.device (valueShape c.dims) c.value
  queryDense : ValidDense c.query.tensor.device (queryShape c.dims) a.queryDense
  keyDense : ValidDense c.query.tensor.device (queryShape c.dims) a.keyDense
  valueDense : ValidDense c.query.tensor.device (valueShape c.dims) a.valueDense
  output : ValidDense c.query.tensor.device (valueShape c.dims) a.output
  fresh : PublicationAllocation (forwardBuffers a) (forwardProtected c a)

structure BackwardAPI (c : Invocation) (a : Allocations) (env : LaunchEnvironment) : Prop where
  launch : LaunchContract c.query.tensor.device env
  incoming : ValidView c.query.tensor.device (valueShape c.dims) c.incoming
  queryDense : ValidDense c.query.tensor.device (queryShape c.dims) a.queryDense
  keyDense : ValidDense c.query.tensor.device (queryShape c.dims) a.keyDense
  valueDense : ValidDense c.query.tensor.device (valueShape c.dims) a.valueDense
  incomingDense : ValidDense c.query.tensor.device (valueShape c.dims) a.incomingDense
  queryGradient : ValidDense c.query.tensor.device (queryShape c.dims) a.queryGradient
  keyGradient : ValidDense c.query.tensor.device (queryShape c.dims) a.keyGradient
  valueGradient : ValidDense c.query.tensor.device (valueShape c.dims) a.valueGradient
  fresh : PublicationAllocation (backwardBuffers a) (backwardProtected c a)

def forward_contract (c : Invocation) (a : Allocations) (env : LaunchEnvironment) (o : Observations)
    (h : InputContract c.dims o (forwardTensors a)) (binding : PointerBinding o (forwardTensors a))
    (api : ForwardAPI c a env) : ForwardContract c a env :=
  ⟨dimensions _ _ _ h,api.launch,api.query,api.key,api.value,api.queryDense,api.keyDense,
    api.valueDense,api.output,⟨api.fresh,forward_pointers c a o h binding⟩⟩

def backward_contract (c : Invocation) (a : Allocations) (env : LaunchEnvironment) (o : Observations)
    (h : InputContract c.dims o (backwardTensors a)) (binding : PointerBinding o (backwardTensors a))
    (api : BackwardAPI c a env) : BackwardContract c a env :=
  ⟨dimensions _ _ _ h,api.launch,api.incoming,api.queryDense,api.keyDense,api.valueDense,
    api.incomingDense,api.queryGradient,api.keyGradient,api.valueGradient,
    ⟨api.fresh,backward_pointers c a o h binding⟩⟩

#axiomsClean check_iff
#axiomsClean dimensions
#axiomsClean aligned
#axiomsClean forward_pointers
#axiomsClean backward_pointers
#specNonCircular InputContract avoiding [publicForward,publicBackward,FP.Structural.exec]
#specNonCircular ForwardAPI avoiding [publicForward,publicBackward,FP.Structural.exec]
#specNonCircular BackwardAPI avoiding [publicForward,publicBackward,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LightningAttention.Checked
