import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperOutputStorage
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedState

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

/-- The allocation origin is the actual state after all input resolves. Thus
freshness includes materialized copies, not just the caller's original handles. -/
theorem prepare_storage (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : FreshAllocationAPI store api)
    (q : Request) (s : S) (hp : RequestPresent store q s) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) :
    ∃ allocationStart,
      store.Extends s allocationStart ∧
      InputsMaterialized store q s p.inputs allocationStart ∧
      AllocationBatch store allocationStart (outputTensors p.outputs) ((prepare api q).run s).state := by
  have hh := h
  unfold prepare at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨v,_,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨input,hi,hh,h₁,_⟩ := bind_success _ _ _ _ hh
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
  have hv : store.Extends s sv := persists_validate store q s
  have hiPreserved : store.Extends sv si := persists_resolveInputs store api hs q sv
  have hiValues : InputsMaterialized store q sv input si :=
    resolveInputs_materialized store api hs q sv (hp.transfer hv) input hi
  have hb : AllocationBatch store si (outputTensors output) so :=
    allocateOutputs_storage store api hs hf q v input si output ho
  rw [hpEq,hfinal]
  exact ⟨si,hv.trans hiPreserved,hiValues.transfer hp hv (store.extends_refl si),
    hb.transfer (persists_width store api hs v input so)⟩

#axiomsClean prepare_storage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
