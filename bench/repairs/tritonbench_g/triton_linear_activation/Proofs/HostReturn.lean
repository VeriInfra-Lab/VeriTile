import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.HostRun
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

def HostMetadata (r : Request) (h : Allocation r caller before) (result : HostResult α)
    (currentDevice : Nat) : Prop :=
  result.returned.storage=h.storage 0 ∧
  result.returned.shape=r.batch.shape r.n ∧
  result.returned.strides=canonicalStrides (r.batch.shape r.n) ∧
  result.returned.dtype=r.mode ∧ result.returned.device=r.device ∧
  result.returned.offset=0 ∧ result.returned.negative=false ∧ result.returned.conjugate=false ∧
  result.returned.storage∉caller ∧ (∀ i,result.returned.objectId≠r.object i) ∧
  result.context=(h.computeTensor 1,if r.hasBias then some (h.computeTensor 2) else none,h.computeTensor 0) ∧
  result.callerSaved=(if r.hasSaved then some (r.argument 3) else none) ∧
  result.currentDevice=currentDevice

theorem Allocation.host_metadata (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (currentDevice : Nat) : HostMetadata r h (h.hostResult a c currentDevice) currentDevice := by
  refine ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,h.fresh 0,?_,rfl,rfl,rfl⟩
  intro i
  cases batch : r.batch <;> simp only [Allocation.hostResult,Allocation.returned,tensor,batch]
  · exact h.objectFresh 0 i
  · exact h.objectFresh 5 i

theorem Allocation.context_off (h : Allocation r caller before) (mode : r.mode=.fp32) :
    h.context=(r.argument 1,if r.hasBias then some (r.argument 2) else none,r.argument 0) := by
  simp only [Allocation.context,Allocation.computeTensor,mode,inputSlot]
  rfl

theorem Allocation.context_amp (h : Allocation r caller before) (mode : r.mode=.autocastFP16)
    (i : Fin 3) : (h.computeTensor i).dtype=.autocastFP16 ∧
      (h.computeTensor i).shape=r.argumentShape (inputSlot i) ∧
      (h.computeTensor i).storage=h.storage (castSlot i) ∧
      (h.computeTensor i).objectId≠r.object (inputSlot i) := by
  simp only [Allocation.computeTensor,mode,tensor]
  exact ⟨trivial,trivial,trivial,h.objectFresh _ _⟩

/-- Local CUDA device-context state update used around the kernel only. -/
structure DeviceState (α : Type) where
  currentDevice : Nat
  payload : α
def deviceCall (owner : Nat) (launch : DeviceState α → DeviceState α) (state : DeviceState α) :=
  let finished := launch ⟨owner,state.payload⟩
  { finished with currentDevice := state.currentDevice }
theorem deviceCall_restores (owner : Nat) (launch : DeviceState α → DeviceState α) (state : DeviceState α) :
    (deviceCall owner launch state).currentDevice=state.currentDevice := rfl
theorem deviceCall_enters (owner : Nat) (launch : DeviceState α → DeviceState α) (state : DeviceState α) :
    (deviceCall owner launch state).payload=(launch ⟨owner,state.payload⟩).payload := rfl

#axiomsClean Allocation.host_metadata
#axiomsClean Allocation.context_off
#axiomsClean Allocation.context_amp
#axiomsClean deviceCall_restores
end
end VeriTile.Bench.Repairs.TritonLinearActivation
