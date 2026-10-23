import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperNamedViews

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

theorem written_role (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (key : AliasMemory.Key) (h : Contract.written f r stride M N row key.1 key.2) :
    ∃ role col, NativeMachineMemory.writeRole role ∧ LayerNormFwd.Address.active f role ∧
      NativeAccess.laneActive N col role ∧
      key=(LayerNormFwd.Address.region r role,LayerNormFwd.Address.offset stride M N row col role) := by
  rcases h with ⟨hf,hregion,hrow⟩ | ⟨hregion,hrow⟩ | ⟨hregion,hoffset⟩ | ⟨hf,hregion,hoffset⟩ |
    ⟨hf,hregion,hrow⟩ | ⟨hd,hm,hx,hregion,hrow⟩ | ⟨hd,hm,hregion,hrow⟩
  · obtain ⟨col,hc,hoffset⟩ := Contract.inRow_iff_col.mp hrow
    exact ⟨.y1,col,trivial,hf,hc,Prod.ext hregion hoffset⟩
  · obtain ⟨col,hc,hoffset⟩ := Contract.inRow_iff_col.mp hrow
    exact ⟨.y,col,trivial,trivial,hc,Prod.ext hregion hoffset⟩
  · exact ⟨.rstd,0,trivial,trivial,trivial,Prod.ext hregion hoffset⟩
  · exact ⟨.mean,0,trivial,hf,trivial,Prod.ext hregion hoffset⟩
  · obtain ⟨col,hc,hoffset⟩ := Contract.inRow_iff_col.mp hrow
    exact ⟨.residualOut,col,trivial,hf,hc,Prod.ext hregion hoffset⟩
  · obtain ⟨col,hc,hoffset⟩ := Contract.inRow_iff_col.mp hrow
    exact ⟨.mask1,col,trivial,⟨hx,hd,hm⟩,hc,Prod.ext hregion hoffset⟩
  · obtain ⟨col,hc,hoffset⟩ := Contract.inRow_iff_col.mp hrow
    exact ⟨.mask,col,trivial,⟨hd,hm⟩,hc,Prod.ext hregion hoffset⟩

theorem write_role_output (p : Prepared) (role : LayerNormFwd.Address.Role)
    (hw : NativeMachineMemory.writeRole role) (tensor : Tensor) (he : roleTensor p role=some tensor) :
    (LayerNormFwd.Address.region preparedRegions role,tensor)∈outputViews p.outputs := by
  cases role <;>
    simp_all [NativeMachineMemory.writeRole,roleTensor,LayerNormFwd.Address.region,
      outputViews,namedOptional,optionTensors]

theorem written_live_output {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) (key : AliasMemory.Key)
    (hw : AliasMemory.Writes (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N key) :
    ∃ output, regionTensor p key.1=some output ∧
      key.2<(preparedAllocations store current p key.1).extent ∧
      (key.1,output)∈outputViews p.outputs := by
  obtain ⟨row,hr,hwrite⟩ := hw
  obtain ⟨role,col,hrole,ha,hl,hkey⟩ := written_role _ _ _ _ _ row key hwrite
  obtain ⟨output,he,_,_,_,_⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  rw [hkey]
  exact ⟨output,by rw [regionTensor_role]; exact he,
    prepared_role_bounds store hm tensor q initial current p args h hv hi ho row col hr role ha hl,
    write_role_output p role hrole output he⟩

/-- FreshWrites is derived from actual allocations, including every resolved
input and every optional output/seed view. Read/read aliases remain allowed. -/
theorem prepared_fresh_writes {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) :
    AliasMemory.FreshWrites (PhysicalDomain.layout (preparedAllocations store current p))
      (AliasMemory.Writes (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N) := by
  obtain ⟨allocationStart,_,hinputs,hbatch⟩ := h.storage
  intro w hw
  obtain ⟨output,houtput,hwbound,homem⟩ := written_live_output store hm tensor q initial current p args h hv hi ho w hw
  have hdout : preparedAllocations store current p w.1=store.descriptor current output := by
    simp [preparedAllocations,houtput]
  refine ⟨hwbound,?_⟩
  intro k hk he
  by_cases hregion : k.1=w.1
  · apply Prod.ext hregion
    change (preparedAllocations store current p k.1).base+k.2*(preparedAllocations store current p k.1).width=
      (preparedAllocations store current p w.1).base+w.2*(preparedAllocations store current p w.1).width at he
    rw [hregion] at he
    exact Nat.mul_right_cancel (preparedAllocations store current p w.1).width_pos (Nat.add_left_cancel he)
  · obtain ⟨value,hvalue,hdvalue⟩ := allocation_live_tensor store current p k.1 k.2 hk
    have hseparate : Separate store current output value := by
      rcases regionTensor_member p k.1 value hvalue with hinput | houtput'
      · have hmem : output∈outputTensors p.outputs := by
          have hmap := List.mem_map_of_mem (f:=Prod.snd) homem
          simpa only [outputViews_values] using hmap
        exact hbatch.prior output hmem value (inputViews_present hinputs k.1 value hinput)
      · exact outputViews_separate hbatch w.1 k.1 output value homem houtput' (Ne.symm hregion)
    have hneq := ByteApart.cells hseparate w.2 k.2
      (by simpa only [hdout] using hwbound) (by simpa only [PhysicalDomain.layout,hdvalue] using hk)
    exact False.elim (hneq (by simpa only [PhysicalDomain.layout,hdout,hdvalue] using he.symm))

#axiomsClean written_role
#axiomsClean write_role_output
#axiomsClean written_live_output
#axiomsClean prepared_fresh_writes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
