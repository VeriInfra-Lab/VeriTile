import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageAction
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedContents
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperObservationFrame

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

structure Placement (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (q : Request) (p : Prepared) (args : LaunchArguments) (current : HostState α Host)
    (c : StorageMachine.Config α block) : Prop where
  flags : c.flags=p.flags q
  strides : c.strides=argumentStrides args
  rows : c.M=p.validated.M
  cols : c.N=p.validated.N
  allocations : c.allocations=preparedAllocations store current p

theorem descriptor_read (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (allocations : RegionName → Descriptor) (s : HostState α Host) (value : Tensor)
    (hp : store.present s value) (hn : value.negative=Bool.false) (region : RegionName)
    (hd : allocations region=store.descriptor s value) (index : List Nat) (hi : Coordinate value index)
    (offset : Nat) (ho : offset=coordinateOffset value.strides index) :
    TypedPhysical.read (PhysicalDomain.layout allocations) s.memory (region,offset)=store.read s value index := by
  have hl : (PhysicalDomain.layout allocations).live (region,offset) := by
    change offset<(allocations region).extent
    rw [hd,ho]
    exact hm.bounds s value hp index hi
  rw [TypedPhysical.read,dif_pos hl,hm.read s value hp hn index hi]
  congr 1
  apply Fin.ext
  change LayerNormFwd.Address.byteAddress (allocations region) offset=_
  rw [hd,ho]

theorem prepared_value_readback
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current native : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : c.allocations=preparedAllocations store current p)
    (hn : native.host=current.host)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role) (hl : NativeAccess.laneActive p.validated.N col role)
    (value : Tensor) (he : roleTensor p role=some value) :
    store.read native value (roleIndex p.validated.M row col role)=
      TypedPhysical.read (PhysicalDomain.layout c.allocations) native.memory
        (LayerNormFwd.Address.region preparedRegions role,
          LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role) := by
  obtain ⟨actual,heq,hp,hpositive,hcoordinate,hoffset⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  have hav : actual=value := Option.some.inj (heq.symm.trans he)
  subst actual
  have hp' := hh.present current native hn.symm value hp
  apply Eq.symm
  apply descriptor_read store hm c.allocations native value hp' hpositive _ _ _ hcoordinate _ hoffset
  rw [hc,preparedAllocation_role store current p role value he]
  exact hh.descriptor current native hn.symm value

theorem prepared_input_original
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : c.allocations=preparedAllocations store current p)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role) (hl : NativeAccess.laneActive p.validated.N col role)
    (hrole : sourceInputRole role) :
    ∃ source, originalRoleTensor q role=some source ∧
      TypedPhysical.read (PhysicalDomain.layout c.allocations) current.memory
        (LayerNormFwd.Address.region preparedRegions role,
          LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role)=
      store.read initial source (roleIndex p.validated.M row col role) := by
  obtain ⟨value,he,_,_,hcoordinate,hoffset⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  obtain ⟨source,hsource,hmat⟩ := role_materialized store tensor q initial current p args h role hrole value he
  refine ⟨source,hsource,?_⟩
  exact (descriptor_read store hm c.allocations current value hmat.present hmat.positive _
    (by rw [hc]; exact preparedAllocation_role store current p role value he) _ hcoordinate _ hoffset).trans
      (hmat.values _)

theorem prior_outside
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : Placement store q p args current c)
    (old : Tensor) (hold : store.present initial old) (byte : Nat)
    (hbyte : byte<(store.descriptor initial old).extent*(store.descriptor initial old).width)
    (i : Address) (haddress : i.val=(store.descriptor initial old).base+byte)
    (k : AliasMemory.Key) (hk : AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N k) :
    (PhysicalDomain.layout c.allocations).address k≠i.val := by
  have hw : AliasMemory.Writes (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N k := by
    simpa only [hc.flags,hc.strides,hc.rows,hc.cols] using hk
  obtain ⟨allocationStart,hprior,_,hbatch⟩ := h.storage
  obtain ⟨output,houtput,hbound,hmember⟩ := written_live_output store hm tensor q initial current p args h hv hi ho k hw
  have homem : output∈outputTensors p.outputs := by
    have hmap := List.mem_map_of_mem (f:=Prod.snd) hmember
    simpa only [outputViews_values] using hmap
  have hseparate := hbatch.prior output homem old (hprior old hold).1
  have hdout : preparedAllocations store current p k.1=store.descriptor current output := by
    simp [preparedAllocations,houtput]
  have hdold := (h.preserved old hold).2.1
  have hout : k.2<(store.descriptor current output).extent := by simpa only [hdout] using hbound
  have hbyte' : byte<(store.descriptor current old).extent*(store.descriptor current old).width := by
    rw [hdold]
    exact hbyte
  have hneq := hseparate (k.2*(store.descriptor current output).width) byte
    (Nat.mul_lt_mul_of_pos_right hout (store.descriptor current output).width_pos) hbyte'
  change (c.allocations k.1).base+k.2*(c.allocations k.1).width≠i.val
  rw [hc.allocations,hdout,haddress]
  simpa only [hdold] using hneq

theorem dispatch_prior_read [Inhabited α]
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : Placement store q p args current c)
    (benchmarks : List (StorageMachine.DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block)
    (old : Tensor) (hold : store.present initial old) (index : List Nat) (hindex : Coordinate old index) :
    store.read {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory} old index=
      store.read initial old index := by
  let native := {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}
  have hp := (h.preserved old hold).1
  have hd := (h.preserved old hold).2.1
  have he : store.read native old index=store.read current old index := by
    apply hr.read current native old hp (hh.present current native rfl old hp)
      (hh.descriptor native current rfl old) _ index hindex
    intro i hlo hhi
    rw [hd] at hlo hhi
    have hbyte : i.val-(store.descriptor initial old).base<
        (store.descriptor initial old).extent*(store.descriptor initial old).width := by omega
    have haddress : i.val=(store.descriptor initial old).base+
        (i.val-(store.descriptor initial old).base) := by omega
    exact StorageMachine.dispatch_frame c benchmarks selected current.memory i
      (prior_outside store hm tensor q initial current p args h hv hi ho c hc old hold _ hbyte i haddress)
  exact he.trans ((h.preserved old hold).2.2 index)

theorem dispatch_fallback_readback [Inhabited α]
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hl : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (hp : RequestPresent store q initial)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : Placement store q p args current c)
    (benchmarks : List (StorageMachine.DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block)
    (row col : Nat) (hr : row<p.validated.M) (hcol : col<p.validated.N) :
    let native := {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}
    store.read native p.inputs.originalX [row,col]=
      TypedPhysical.read (PhysicalDomain.layout c.allocations) native.memory
        (preparedRegions.x,row*(argumentStrides args).x+col) := by
  have hindex := coordinate_matrix q.x p.validated.M p.validated.N row col hv.x.1 hr hcol
  have hprior := dispatch_prior_read store hm hh hl tensor q initial current p args h hv hi ho c hc
    benchmarks selected q.x hp.x [row,col] hindex
  obtain ⟨source,hsource,hvalue⟩ := prepared_input_original store hm tensor q initial current p args h hv hi ho
    c hc.allocations row col hr .x trivial hcol trivial
  have hs : q.x=source := Option.some.inj hsource
  rw [←hs] at hvalue
  have hframe := StorageMachine.dispatch_input_read c benchmarks selected current.memory preparedRegions.x
    (by simp [StorageMachine.inputRegion]) (row*(argumentStrides args).x+col)
  dsimp only
  rw [h.inputs.original]
  exact hprior.trans (hvalue.symm.trans hframe.symm)

#axiomsClean descriptor_read
#axiomsClean prepared_value_readback
#axiomsClean prepared_input_original
#axiomsClean prior_outside
#axiomsClean dispatch_prior_read
#axiomsClean dispatch_fallback_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
