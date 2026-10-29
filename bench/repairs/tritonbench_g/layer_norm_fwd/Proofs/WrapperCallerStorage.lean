import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageDomain

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000

/-- The caller's protected bytes belong to tensors live before this wrapper
call. This permits aliased views and arbitrary byte offsets, and states no
kernel-output or successful-execution equation. -/
def CallerStorage {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (initial : HostState α Host) (caller : Address → Prop) : Prop :=
  ∀ i, caller i → ∃ tensor byte, store.present initial tensor ∧
    byte<(store.descriptor initial tensor).extent*(store.descriptor initial tensor).width ∧
    i.val=(store.descriptor initial tensor).base+byte

/-- Every actual write is disjoint from all pre-wrapper caller bytes. This is
derived from the allocation sequence, including resolved-copy allocations,
rather than supplied as a whole execution frame assumption. -/
theorem prepared_caller_outside {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) (hc : CallerStorage store initial caller)
    (i : Address) (hcaller : caller i) (row : Fin p.validated.M)
    (region : RegionName) (offset : Nat)
    (hw : Contract.written (p.flags q) preparedRegions (argumentStrides args)
      p.validated.M p.validated.N row.val region offset) :
    (PhysicalDomain.layout (preparedAllocations store current p)).address (region,offset)≠i.val := by
  obtain ⟨old,byte,hold,hbyte,haddress⟩ := hc i hcaller
  obtain ⟨allocationStart,hprior,_,hbatch⟩ := h.storage
  obtain ⟨output,houtput,hbound,hmember⟩ :=
    written_live_output store hm tensor q initial current p args h hv hi ho (region,offset)
      ⟨row.val,row.isLt,hw⟩
  have homem : output∈outputTensors p.outputs := by
    have hm := List.mem_map_of_mem (f:=Prod.snd) hmember
    simpa only [outputViews_values] using hm
  have hseparate := hbatch.prior output homem old (hprior old hold).1
  have hdout : preparedAllocations store current p region=store.descriptor current output := by
    simp [preparedAllocations,houtput]
  have hdold := (h.preserved old hold).2.1
  have hout : offset<(store.descriptor current output).extent := by simpa only [hdout] using hbound
  have hbyte' : byte<(store.descriptor current old).extent*(store.descriptor current old).width := by
    rw [hdold]
    exact hbyte
  have hneq := hseparate (offset*(store.descriptor current output).width) byte
    (Nat.mul_lt_mul_of_pos_right hout (store.descriptor current output).width_pos) hbyte'
  change (preparedAllocations store current p region).base+
    offset*(preparedAllocations store current p region).width≠i.val
  rw [hdout,haddress]
  simpa only [hdold] using hneq

#axiomsClean prepared_caller_outside
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
