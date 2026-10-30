import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedInputs
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperOutputAllocations
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperOrder

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

theorem prepare_allocated (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) :
    OutputsAllocated store api q p.validated p.inputs p.outputs ((prepare api q).run s).state := by
  have hh := h
  unfold prepare at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨v,_,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨input,_,hh,h₁,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨output,ho,hh,h₂,_⟩ := bind_success _ _ _ _ hh
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
  have hoValues : OutputsAllocated store api q v input output so :=
    allocateOutputs_allocated store api hc q v input si output ho
  have hf : store.Extends so sf := persists_width store api hc v input so
  rw [hpEq,hfinal]
  exact hoValues.transfer hf

/-- The actual successful pre-launch state: resolved logical input values,
live output descriptors with source allocation metadata/freshness, and the
late width assertion. No kernel result or benchmark success is assumed. -/
theorem prepare_state (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) (s : S) (hp : RequestPresent store q s) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) :
    InputsMaterialized store q s p.inputs ((prepare api q).run s).state ∧
    OutputsAllocated store api q p.validated p.inputs p.outputs ((prepare api q).run s).state ∧
    p.validated.N≤p.block :=
  ⟨prepare_materialized store api hc q s hp p h,prepare_allocated store api hc q s p h,
    (prepare_success api q s p h).1⟩

#axiomsClean prepare_allocated
#axiomsClean prepare_state
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
