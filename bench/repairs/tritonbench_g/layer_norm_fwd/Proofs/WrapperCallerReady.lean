import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperCallerAllocation

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

theorem prepare_caller_storage (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : CallerAllocationAPI store api caller)
    (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) :
    ∃ allocationStart, CallerBatch store caller allocationStart (outputTensors p.outputs)
      ((prepare api q).run s).state := by
  unfold prepare at h
  dsimp only [Bind.bind] at h
  obtain ⟨v,_,h,h₀,_⟩ := bind_success _ _ _ _ h
  obtain ⟨input,_,h,h₁,_⟩ := bind_success _ _ _ _ h
  obtain ⟨output,ho,h,h₂,_⟩ := bind_success _ _ _ _ h
  obtain ⟨block,_,h,h₃,_⟩ := bind_success _ _ _ _ h
  have hp : p=⟨v,input,output,block⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  let sv := ((validate (world:=world) q).run s).state
  let si := ((resolveInputs api q).run sv).state
  let so := ((allocateOutputs api q v input).run si).state
  let sf := ((width api v input).run so).state
  have hfinal : ((prepare api q).run s).state=sf := by
    unfold prepare
    dsimp only [Bind.bind]
    rw [h₀,h₁,h₂,h₃]
    rfl
  have hb : CallerBatch store caller si (outputTensors output) so :=
    allocateOutputs_caller_storage store api hs hf q v input si output ho
  rw [hp,hfinal]
  exact ⟨si,hb.transfer (persists_width store api hs v input so)⟩

theorem prefix_caller_storage (store : TensorStore S Descriptor Value)
    (tensor : TensorAPI world) (device : DispatchAPI world)
    (hs : StorageAPI store tensor) (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (q : Request) (initial : S) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args) :
    ∃ allocationStart, CallerBatch store caller allocationStart (outputTensors p.outputs)
      (launchState tensor device q initial p args) := by
  obtain ⟨allocationStart,h⟩ := prepare_caller_storage store tensor hs hc q initial p route.prepared
  exact ⟨allocationStart,h.transfer (setup_preserves store tensor device hd q initial p args)⟩

theorem prepared_full_caller_outside {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (WrapperNative.world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current allocationStart : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (hc : CallerBatch store caller allocationStart (outputTensors p.outputs) current)
    (i : Address) (hcaller : caller i) (row : Fin p.validated.M) (region : RegionName) (offset : Nat)
    (hw : Contract.written (p.flags q) preparedRegions (argumentStrides args)
      p.validated.M p.validated.N row.val region offset) :
    (PhysicalDomain.layout (preparedAllocations store current p)).address (region,offset)≠i.val := by
  obtain ⟨output,houtput,hbound,hmember⟩ :=
    written_live_output store hm tensor q initial current p args h hv hi ho (region,offset)
      ⟨row.val,row.isLt,hw⟩
  have homem : output∈outputTensors p.outputs := by
    have hmap := List.mem_map_of_mem (f:=Prod.snd) hmember
    simpa only [outputViews_values] using hmap
  have hd : preparedAllocations store current p region=store.descriptor current output := by
    simp [preparedAllocations,houtput]
  have hout : offset<(store.descriptor current output).extent := by simpa only [hd] using hbound
  have hneq := hc.outside output homem i hcaller (offset*(store.descriptor current output).width)
    (Nat.mul_lt_mul_of_pos_right hout (store.descriptor current output).width_pos)
  simpa only [PhysicalDomain.layout,hd,AliasMemory.viewAddress,LayerNormFwd.Address.byteAddress] using hneq

/-- Complete local launch obligations from the actual admitted host prefix,
with caller freshness covering the whole protected backing storage. -/
theorem prefix_ready_full_caller {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (WrapperNative.world (α:=α) (Host:=Host) caller))
    (device : DispatchAPI (WrapperNative.world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial)
    (p : Prepared) (args : LaunchArguments) (route : LaunchPrefix tensor device q initial p args)
    (hprofile : SelectedStorage q p) :
    PreparedReady store args (launchState tensor device q initial p args) caller := by
  have h := prefix_launch_state store tensor device hs hf hd q initial hp p args route
  obtain ⟨hv,hi,ho⟩ := prefix_layout tensor device hl q initial p args route
  obtain ⟨allocationStart,hbatch⟩ := prefix_caller_storage store tensor device hs hc hd q initial p args route
  have haccepted := prefix_admission tensor device ha q initial p args route hprofile
  have hstorage := prepared_storage_domain store hm tensor q initial _ p args h hv hi ho hprofile
  have houtside := prepared_full_caller_outside store hm tensor q initial _ allocationStart p args h hv hi ho hbatch
  have hgeometry := launch_geometry store hm tensor q initial _ p args h hv ho preparedRegions prepared_output_names
  refine ⟨?_,?_,?_,?_⟩
  · simpa only [h.arguments.prepared] using haccepted
  · simpa only [h.arguments.prepared,h.arguments.request] using hstorage
  · intro i hi row region offset hw
    have hrow : row.val<p.validated.M := by simpa only [h.arguments.prepared] using row.isLt
    have hw' : Contract.written (p.flags q) preparedRegions (argumentStrides args)
        p.validated.M p.validated.N row.val region offset := by
      simpa only [h.arguments.prepared,h.arguments.request] using hw
    simpa only [h.arguments.prepared] using houtside i hi ⟨row.val,hrow⟩ region offset hw'
  · simpa only [h.arguments.prepared,h.arguments.request] using hgeometry

#axiomsClean prepare_caller_storage
#axiomsClean prefix_caller_storage
#axiomsClean prepared_full_caller_outside
#axiomsClean prefix_ready_full_caller
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
