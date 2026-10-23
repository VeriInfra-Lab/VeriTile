import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperResolvedInputs

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

theorem OptionPresent.transfer {store : TensorStore S Descriptor Value} {s t : S} {tensor : Option Tensor}
    (h : OptionPresent store s tensor) (he : store.Extends s t) : OptionPresent store t tensor := by
  cases tensor
  · trivial
  · exact (he _ h).1

theorem RequestPresent.transfer {store : TensorStore S Descriptor Value} {q : Request} {s t : S}
    (h : RequestPresent store q s) (he : store.Extends s t) : RequestPresent store q t :=
  ⟨(he q.x h.x).1,(he q.weight h.weight).1,h.bias.transfer he,h.residual.transfer he,h.x1.transfer he,
    h.weight1.transfer he,h.bias1.transfer he,h.rowscale.transfer he⟩

theorem InputsMaterialized.transfer {store : TensorStore S Descriptor Value} {q : Request} {input : Inputs}
    {initial before after final : S} (h : InputsMaterialized store q before input after)
    (hp : RequestPresent store q initial) (hb : store.Extends initial before) (ha : store.Extends after final) :
    InputsMaterialized store q initial input final :=
  ⟨h.original,h.x.transfer hp.x hb ha,h.weight.transfer hp.weight hb ha,
    h.bias.transfer hp.bias hb ha,h.residual.transfer hp.residual hb ha,h.x1.transfer hp.x1 hb ha,
    h.weight1.transfer hp.weight1 hb ha,h.bias1.transfer hp.bias1 hb ha,h.rowscale.transfer hp.rowscale hb ha⟩

/-- The successful prepare state contains all actual resolved inputs, with
values from before the wrapper call. Validation, every output allocation and
the late width check run on their real predecessor states. -/
theorem prepare_materialized (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) (s : S) (hp : RequestPresent store q s) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) :
    InputsMaterialized store q s p.inputs ((prepare api q).run s).state := by
  have hh := h
  unfold prepare at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨v,_,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨input,hi,hh,h₁,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨output,_,hh,h₂,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨block,_,hh,h₃,_⟩ := bind_success _ _ _ _ hh
  have hpEq : p=⟨v,input,output,block⟩ := by simpa [Pure.pure,Action.pure] using hh.symm
  let sv := ((validate (world:=world) q).run s).state
  let si := ((resolveInputs api q).run sv).state
  let so := ((allocateOutputs api q v input).run si).state
  let sf := ((width api v input).run so).state
  have hfinal : ((prepare api q).run s).state=sf := by
    unfold prepare
    dsimp only [Bind.bind]
    rw [h₀,h₁,h₂,h₃]
    rfl
  have hv : store.Extends s sv := persists_validate store q s
  have ho : store.Extends si so := persists_allocateOutputs store api hc q v input si
  have hf : store.Extends so sf := persists_width store api hc v input so
  have hiValues : InputsMaterialized store q sv input si :=
    resolveInputs_materialized store api hc q sv (hp.transfer hv) input hi
  rw [hpEq,hfinal]
  exact hiValues.transfer hp hv (ho.trans hf)

#axiomsClean RequestPresent.transfer
#axiomsClean InputsMaterialized.transfer
#axiomsClean prepare_materialized
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
