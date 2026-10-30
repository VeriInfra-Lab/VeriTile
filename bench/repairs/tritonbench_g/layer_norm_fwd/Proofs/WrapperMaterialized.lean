import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPrefixStorage

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 2000000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

structure Materialized (store : TensorStore S Descriptor Value) (original : Tensor) (initial : S)
    (output : Tensor) (current : S) : Prop where
  present : store.present current output
  shape : output.shape=original.shape
  dtype : output.dtype=original.dtype
  positive : output.negative=Bool.false
  values : ∀ index, store.read current output index=store.read initial original index

def OptionPresent (store : TensorStore S Descriptor Value) (s : S) : Option Tensor → Prop
  | none => True
  | some tensor => store.present s tensor

def OptionMaterialized (store : TensorStore S Descriptor Value) (original : Option Tensor) (initial : S)
    (output : Option Tensor) (current : S) : Prop :=
  match original,output with
  | none,none => True
  | some original,some output => Materialized store original initial output current
  | _,_ => False

theorem Materialized.transfer {store : TensorStore S Descriptor Value} {original output : Tensor}
    {initial before after final : S} (h : Materialized store original before output after)
    (hp : store.present initial original) (hb : store.Extends initial before) (ha : store.Extends after final) :
    Materialized store original initial output final := by
  obtain ⟨hout,_,hvalues⟩ := ha output h.present
  refine ⟨hout,h.shape,h.dtype,h.positive,?_⟩
  intro index
  exact (hvalues index).trans ((h.values index).trans ((hb original hp).2.2 index))

theorem OptionMaterialized.transfer {store : TensorStore S Descriptor Value} {original output : Option Tensor}
    {initial before after final : S} (h : OptionMaterialized store original before output after)
    (hp : OptionPresent store initial original) (hb : store.Extends initial before) (ha : store.Extends after final) :
    OptionMaterialized store original initial output final := by
  cases original <;> cases output
  · trivial
  · exact False.elim h
  · exact False.elim h
  · exact Materialized.transfer h hp hb ha

theorem call_success {α : Type} (stage : Stage) (operation : Local world α) (s : S) (output : α)
    (h : ((call stage operation).run s).result=.ok output) : (operation.run s).2=.ok output := by
  cases he : (operation.run s).2 with
  | error code => simp [call,he,Except.mapError] at h
  | ok value =>
    have hv : value=output := by simpa [call,he,Except.mapError] using h
    rw [hv]

theorem resolve_materialized (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (target : Nat) (role : Role) (input : Tensor) (s : S) (output : Tensor)
    (h : ((materialize api target role input).run s).result=.ok output) :
    Materialized store input s output ((materialize api target role input).run s).state := by
  unfold materialize at h ⊢
  dsimp only [Bind.bind] at h ⊢
  obtain ⟨resolved,hr,ht,hstate,_⟩ := bind_success _ _ _ _ h
  have hr' := call_success _ _ _ _ hr
  have ht' := call_success _ _ _ _ ht
  obtain ⟨hrshape,hrdtype,_,hrneg⟩ := api.resolve_metadata role input s resolved hr'
  obtain ⟨htshape,htdtype,_,htneg⟩ := api.transfer_metadata target role resolved _ output ht'
  rw [hstate]
  refine ⟨hc.transfer_present target role resolved _ output ht',htshape.trans hrshape,
    htdtype.trans hrdtype,htneg.trans hrneg,?_⟩
  intro index
  rw [hc.read]
  exact (api.transfer_values target role resolved _ output ht' index).trans
    (api.resolve_values role input s resolved hr' index)

theorem resolveOption_materialized (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (target : Nat) (role : Role) (input : Option Tensor) (s : S) (output : Option Tensor)
    (h : ((resolveOption api target role input).run s).result=.ok output) :
    OptionMaterialized store input s output ((resolveOption api target role input).run s).state := by
  cases input with
  | none =>
    have ho : output=none := by simpa [resolveOption,Pure.pure,Action.pure] using h.symm
    rw [ho]
    trivial
  | some tensor =>
    unfold resolveOption at h ⊢
    dsimp only [Bind.bind] at h ⊢
    obtain ⟨value,hvalue,hrest,hstate,_⟩ := bind_success _ _ _ _ h
    have ho : output=some value := by simpa [Pure.pure,Action.pure] using hrest.symm
    rw [ho,hstate]
    exact resolve_materialized store api hc target role tensor s value hvalue

#axiomsClean Materialized.transfer
#axiomsClean OptionMaterialized.transfer
#axiomsClean call_success
#axiomsClean resolve_materialized
#axiomsClean resolveOption_materialized
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
