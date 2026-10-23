import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperMaterialized

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 2000000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

/-- Actual allocation metadata, descriptor presence and object freshness.
Random/uninitialized contents are deliberately unconstrained here. -/
structure Allocated (store : TensorStore S Descriptor Value) (request : Allocation)
    (output : Tensor) (current : S) : Prop where
  present : store.present current output
  shape : output.shape=request.shape
  dtype : output.dtype=request.dtype
  device : output.device=request.device
  positive : output.negative=Bool.false
  fresh : ∀ old ∈ request.used, output.object≠old

def OptionAllocated (store : TensorStore S Descriptor Value) (condition : Bool) (request : Allocation)
    (output : Option Tensor) (current : S) : Prop :=
  match condition,output with
  | Bool.false,none => True
  | Bool.true,some tensor => Allocated store request tensor current
  | _,_ => False

theorem Allocated.transfer {store : TensorStore S Descriptor Value} {request : Allocation} {output : Tensor}
    {s t : S} (h : Allocated store request output s) (he : store.Extends s t) : Allocated store request output t :=
  ⟨(he output h.present).1,h.shape,h.dtype,h.device,h.positive,h.fresh⟩

theorem OptionAllocated.transfer {store : TensorStore S Descriptor Value} {request : Allocation}
    {condition : Bool} {output : Option Tensor} {s t : S}
    (h : OptionAllocated store condition request output s) (he : store.Extends s t) :
    OptionAllocated store condition request output t := by
  cases condition <;> cases output
  · trivial
  · exact False.elim h
  · exact False.elim h
  · exact Allocated.transfer h he

theorem allocate_allocated (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (role : Role) (request : Allocation) (s : S) (output : Tensor)
    (h : ((call (.allocate role) (api.allocate role request)).run s).result=.ok output) :
    Allocated store request output ((call (.allocate role) (api.allocate role request)).run s).state := by
  have hs := call_success (.allocate role) (api.allocate role request) s output h
  obtain ⟨hshape,hdtype,hdevice,hneg,hfresh⟩ := api.allocate_metadata role request s output hs
  exact ⟨hc.allocate_present role request s output hs,hshape,hdtype,hdevice,hneg,hfresh⟩

theorem allocateOptional_allocated (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (role : Role) (condition : Bool) (request : Allocation) (assertStride : Bool)
    (s : S) (output : Option Tensor)
    (h : ((allocateOptional api role condition request assertStride).run s).result=.ok output) :
    OptionAllocated store condition request output ((allocateOptional api role condition request assertStride).run s).state := by
  cases condition
  · have ho : output=none := by simpa [allocateOptional,Pure.pure,Action.pure] using h.symm
    rw [ho]
    trivial
  · unfold allocateOptional at h ⊢
    simp only [Bool.true_eq,ite_true,Bind.bind] at h ⊢
    obtain ⟨tensor,htensor,hrest,hstate,_⟩ := bind_success _ _ _ _ h
    have hv := allocate_allocated store api hc role request s tensor htensor
    cases hs : assertStride
    · simp only [hs,Bool.false_eq_true,ite_false] at hrest hstate ⊢
      obtain ⟨_,_,hfinal,hstate2,_⟩ := bind_success _ _ _ _ hrest
      have ho : output=some tensor := by simpa [Pure.pure,Action.pure] using hfinal.symm
      rw [ho,hstate,hstate2]
      exact hv
    · simp only [hs,ite_true] at hrest hstate ⊢
      obtain ⟨_,_,hfinal,hstate2,_⟩ := bind_success _ _ _ _ hrest
      have ho : output=some tensor := by simpa [Pure.pure,Action.pure] using hfinal.symm
      rw [ho,hstate,hstate2]
      exact hv.transfer (persists_checkStride store role tensor _)

#axiomsClean Allocated.transfer
#axiomsClean OptionAllocated.transfer
#axiomsClean allocate_allocated
#axiomsClean allocateOptional_allocated
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
