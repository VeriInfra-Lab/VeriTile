import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperRoleReadback

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

/-- Exact argument/descriptor projections; the concrete prepared launch proves
these by construction. No memory value or execution result is included. -/
structure PreparedPlacement (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (q : Request) (p : Prepared) (args : LaunchArguments) (current : HostState α Host)
    (c : Config α block) : Prop where
  flags : c.flags=p.flags q
  regions : c.regions=preparedRegions
  strides : c.strides=argumentStrides args
  rows : c.M=p.validated.M
  cols : c.N=p.validated.N
  allocations : c.allocations=preparedAllocations store current p

theorem preparedLaunch_placement
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α)
    (args : LaunchArguments) (current : HostState α Host)
    (h : PreparedReady store args current caller)
    (benchmarks : List (DispatchEvent args.prepared.block)) (selected : Schedule args.prepared.block) :
    PreparedPlacement store args.request args.prepared args current
      (preparedLaunch store ops encoding args current h benchmarks selected).config :=
  ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

/-- Disjointness also covers old tensors whose handles are not kernel pointer
arguments, such as the original lazy-negative X retained for fallback. -/
theorem prepared_prior_outside
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : Config α block) (hc : PreparedPlacement store q p args current c)
    (old : Tensor) (hold : store.present initial old) (byte : Nat)
    (hbyte : byte<(store.descriptor initial old).extent*(store.descriptor initial old).width)
    (i : Address) (haddress : i.val=(store.descriptor initial old).base+byte)
    (row : Fin c.M) : Outside c row i := by
  intro region offset hw
  have hrow : row.val<p.validated.M := by simpa only [hc.rows] using row.isLt
  have hw' : Contract.written (p.flags q) preparedRegions (argumentStrides args)
      p.validated.M p.validated.N row.val region offset := by
    simpa only [hc.flags,hc.regions,hc.strides,hc.rows,hc.cols] using hw
  obtain ⟨allocationStart,hprior,_,hbatch⟩ := h.storage
  obtain ⟨output,houtput,hbound,hmember⟩ :=
    written_live_output store hm tensor q initial current p args h hv hi ho (region,offset)
      ⟨row.val,hrow,hw'⟩
  have homem : output∈outputTensors p.outputs := by
    have hmap := List.mem_map_of_mem (f:=Prod.snd) hmember
    simpa only [outputViews_values] using hmap
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
  change (c.allocations region).base+offset*(c.allocations region).width≠i.val
  rw [hc.allocations,hdout,haddress]
  simpa only [hdold] using hneq

/-- Ordinary logical-read locality, including lazy-negative tensors. A fixed
view's logical read depends only on its descriptor and underlying cells.
This local tensor contract contains no kernel or whole-output equation. -/
structure LogicalReadLocal (store : TensorStore (HostState α Host) Descriptor (Cell α)) : Prop where
  read : ∀ s t tensor, store.present s tensor → store.present t tensor →
    store.descriptor t tensor=store.descriptor s tensor →
    (∀ i : Address, (store.descriptor s tensor).base ≤ i.val →
      i.val<(store.descriptor s tensor).base+(store.descriptor s tensor).extent*(store.descriptor s tensor).width →
      t.memory i=s.memory i) →
    ∀ index, Coordinate tensor index → store.read t tensor index=store.read s tensor index

/-- Native benchmark and selected launches preserve every logical coordinate
of every old tensor, even if that tensor is an aliased or lazy-negative view
not passed to the kernel. -/
theorem dispatch_prior_read [Inhabited α]
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : Config α block) (hc : PreparedPlacement store q p args current c)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block)
    (old : Tensor) (hold : store.present initial old) (index : List Nat) (hindex : Coordinate old index) :
    store.read {current with memory:=(dispatch c benchmarks selected current.memory).memory} old index=
      store.read initial old index := by
  let native := {current with memory:=(dispatch c benchmarks selected current.memory).memory}
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
    exact dispatch_frame c benchmarks selected current.memory i
      (prepared_prior_outside store hm tensor q initial current p args h hv hi ho c hc old hold _ hbyte i haddress)
  exact he.trans ((h.preserved old hold).2.2 index)

#axiomsClean preparedLaunch_placement
#axiomsClean prepared_prior_outside
#axiomsClean dispatch_prior_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
