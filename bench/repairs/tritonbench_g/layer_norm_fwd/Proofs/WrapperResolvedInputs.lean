import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperMaterialized

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

structure RequestPresent (store : TensorStore S Descriptor Value) (q : Request) (s : S) : Prop where
  x : store.present s q.x
  weight : store.present s q.weight
  bias : OptionPresent store s q.bias
  residual : OptionPresent store s q.residual
  x1 : OptionPresent store s q.x1
  weight1 : OptionPresent store s q.weight1
  bias1 : OptionPresent store s q.bias1
  rowscale : OptionPresent store s q.rowscale

structure InputsMaterialized (store : TensorStore S Descriptor Value) (q : Request) (initial : S)
    (input : Inputs) (current : S) : Prop where
  original : input.originalX=q.x
  x : Materialized store q.x initial input.x current
  weight : Materialized store q.weight initial input.weight current
  bias : OptionMaterialized store q.bias initial input.bias current
  residual : OptionMaterialized store q.residual initial input.residual current
  x1 : OptionMaterialized store q.x1 initial input.x1 current
  weight1 : OptionMaterialized store q.weight1 initial input.weight1 current
  bias1 : OptionMaterialized store q.bias1 initial input.bias1 current
  rowscale : OptionMaterialized store q.rowscale initial input.rowscale current

/-- All resolved handles are live in the final resolution state, with their
original logical coordinates. Every intervening resolve executes on and
preserves its actual predecessor state, including earlier materializations. -/
theorem resolveInputs_materialized (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) (s : S) (hp : RequestPresent store q s) (input : Inputs)
    (h : ((resolveInputs api q).run s).result=.ok input) :
    InputsMaterialized store q s input ((resolveInputs api q).run s).state := by
  have hh := h
  unfold resolveInputs at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨_,_,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨x,hx,hh,h₁,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨w,hw,hh,h₂,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨b,hb,hh,h₃,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨r,hr,hh,h₄,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨x1,hx1,hh,h₅,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨w1,hw1,hh,h₆,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨b1,hb1,hh,h₇,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨rs,hrs,hh,h₈,_⟩ := bind_success _ _ _ _ hh
  have hi : input=⟨q.x,x,w,b,r,x1,w1,b1,rs⟩ := by simpa [Pure.pure,Action.pure] using hh.symm
  let sx := ((materialize api q.x.device .x q.x).run s).state
  let sw := ((materialize api q.x.device .weight q.weight).run sx).state
  let sb := ((resolveOption api q.x.device .bias q.bias).run sw).state
  let sr := ((resolveOption api q.x.device .residual q.residual).run sb).state
  let sx1 := ((resolveOption api q.x.device .x1 q.x1).run sr).state
  let sw1 := ((resolveOption api q.x.device .weight1 q.weight1).run sx1).state
  let sb1 := ((resolveOption api q.x.device .bias1 q.bias1).run sw1).state
  let srs := ((resolveOption api q.x.device .rowscale q.rowscale).run sb1).state
  have hfinal : ((resolveInputs api q).run s).state=srs := by
    unfold resolveInputs
    dsimp only [Bind.bind]
    rw [h₀,h₁,h₂,h₃,h₄,h₅,h₆,h₇,h₈]
    rfl
  have ex : store.Extends s sx := persists_materialize store api hc q.x.device .x q.x s
  have ew : store.Extends sx sw := persists_materialize store api hc q.x.device .weight q.weight sx
  have eb : store.Extends sw sb := persists_resolveOption store api hc q.x.device .bias q.bias sw
  have er : store.Extends sb sr := persists_resolveOption store api hc q.x.device .residual q.residual sb
  have ex1 : store.Extends sr sx1 := persists_resolveOption store api hc q.x.device .x1 q.x1 sr
  have ew1 : store.Extends sx1 sw1 := persists_resolveOption store api hc q.x.device .weight1 q.weight1 sx1
  have eb1 : store.Extends sw1 sb1 := persists_resolveOption store api hc q.x.device .bias1 q.bias1 sw1
  have ers : store.Extends sb1 srs := persists_resolveOption store api hc q.x.device .rowscale q.rowscale sb1
  have vx : Materialized store q.x s x sx := resolve_materialized store api hc q.x.device .x q.x s x hx
  have vw : Materialized store q.weight sx w sw := resolve_materialized store api hc q.x.device .weight q.weight sx w hw
  have vb : OptionMaterialized store q.bias sw b sb := resolveOption_materialized store api hc q.x.device .bias q.bias sw b hb
  have vr : OptionMaterialized store q.residual sb r sr := resolveOption_materialized store api hc q.x.device .residual q.residual sb r hr
  have vx1 : OptionMaterialized store q.x1 sr x1 sx1 := resolveOption_materialized store api hc q.x.device .x1 q.x1 sr x1 hx1
  have vw1 : OptionMaterialized store q.weight1 sx1 w1 sw1 := resolveOption_materialized store api hc q.x.device .weight1 q.weight1 sx1 w1 hw1
  have vb1 : OptionMaterialized store q.bias1 sw1 b1 sb1 := resolveOption_materialized store api hc q.x.device .bias1 q.bias1 sw1 b1 hb1
  have vrs : OptionMaterialized store q.rowscale sb1 rs srs := resolveOption_materialized store api hc q.x.device .rowscale q.rowscale sb1 rs hrs
  rw [hi,hfinal]
  refine ⟨rfl,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact vx.transfer hp.x (store.extends_refl s) (ew.trans (eb.trans (er.trans (ex1.trans (ew1.trans (eb1.trans ers))))))
  · exact vw.transfer hp.weight ex (eb.trans (er.trans (ex1.trans (ew1.trans (eb1.trans ers)))))
  · exact vb.transfer hp.bias (ex.trans ew) (er.trans (ex1.trans (ew1.trans (eb1.trans ers))))
  · exact vr.transfer hp.residual (ex.trans (ew.trans eb)) (ex1.trans (ew1.trans (eb1.trans ers)))
  · exact vx1.transfer hp.x1 (ex.trans (ew.trans (eb.trans er))) (ew1.trans (eb1.trans ers))
  · exact vw1.transfer hp.weight1 (ex.trans (ew.trans (eb.trans (er.trans ex1)))) (eb1.trans ers)
  · exact vb1.transfer hp.bias1 (ex.trans (ew.trans (eb.trans (er.trans (ex1.trans ew1))))) ers
  · exact vrs.transfer hp.rowscale (ex.trans (ew.trans (eb.trans (er.trans (ex1.trans (ew1.trans eb1)))))) (store.extends_refl srs)

#axiomsClean resolveInputs_materialized
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
