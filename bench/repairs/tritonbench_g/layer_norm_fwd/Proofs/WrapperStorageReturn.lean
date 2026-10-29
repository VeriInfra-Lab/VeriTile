import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageSuffix
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageNumerics
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStreamLaunch

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

/-- Concrete full-source precision postcondition. All objects refer to the
actual wrapper prefix, selected compiler result, current benchmark memory and
returned tensor coordinates. The caller supplies none of these equations. -/
def ReturnPost (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host) (q : Request) (initial : HostState α Host)
    (result : Returned) (p : Prepared) (args : LaunchArguments) (template : State α) : Prop :=
  let current := launchState tensor device q initial p args
  let final := ((surround tensor device (kernel (caller:=caller) store primitives casts selection) q).run initial).state
  ∃ (ready : PreparedReady store args current caller)
    (produced : StorageCompiler.ProducedDispatch selection.compiler args current.host),
    selection.select args current.host=.ok produced ∧
    selection.compiler.compile (selection.compiler.key args produced.selected.warps) current.host=.ok produced.selected.emission ∧
    WrapperStream.LaunchReady streams q p current ∧
    (∀ (row col : Nat), row<p.validated.M → ∀ (role : LayerNormFwd.Address.Role),
      LayerNormFwd.Address.active (p.flags q) role → NativeAccess.laneActive p.validated.N col role →
      ¬NativeMachineMemory.writeRole role → ∃ value, roleTensor p role=some value ∧
        value.device=streams.activeOwner current ∧ streams.ready value (streams.activeStream current)) ∧
    ((kernel (caller:=caller) store primitives casts selection args).run current)=
      (action store primitives casts args current ready produced).run current ∧
    let c := config store primitives casts args current ready
    (StorageMachine.history c (compiledBenchmarks produced) current.memory).error=none ∧
    final.memory=(StorageMachine.dispatch c (compiledBenchmarks produced) produced.selected.artifact current.memory).memory ∧
    let observed := returnedObserve store final args.prepared
      (HostReturn.seedCount (args.prepared.flags args.request) args.prepared.validated.M) result
    observed=produced.selected.artifact.observe c.ops c.casts preparedRegions c.strides c.M c.eps c.p
      (TypedPhysical.pull (PhysicalDomain.layout c.allocations) current.memory template) ∧
    observed=StorageCompiler.reference c.profile c.ops c.casts produced.selected.artifact.graph.schedule
      c.flags preparedRegions c.strides c.M c.N c.eps c.p
      (TypedPhysical.pull (PhysicalDomain.layout c.allocations) current.memory template) ∧
    (∀ (row col : Nat), row<p.validated.M → ∀ (role : LayerNormFwd.Address.Role),
      LayerNormFwd.Address.active (p.flags q) role → NativeAccess.laneActive p.validated.N col role →
      sourceInputRole role → ∃ source, originalRoleTensor q role=some source ∧
        TypedPhysical.read (PhysicalDomain.layout c.allocations) current.memory
          (LayerNormFwd.Address.region preparedRegions role,
            LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role)=
        store.read initial source (roleIndex p.validated.M row col role)) ∧
    (∀ (row : Fin c.M) (destination : Compiler.Destination),
      (produced.selected.emission.stores destination).map
        (StorageCompiler.Operand.eval c.ops c.casts (StorageMachine.snapshot c current.memory)
          c.M row.val c.eps c.p)=Compiler.storeCoordinate c.flags observed row destination) ∧
    (∀ i, caller i → final.memory i=initial.memory i)

theorem route_return
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hmem : SuffixMemory device)
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial) (hprofile : DefaultInputs q)
    (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device (kernel (caller:=caller) store primitives casts selection) q initial result p args)
    (template : State α) : ReturnPost store tensor device streams primitives casts selection q initial result p args template := by
  have hargs := arguments_bound device q p _ args route.arguments
  have hprepared := hargs.prepared
  have hrequest := hargs.request
  subst p
  subst q
  let current := launchState tensor device args.request initial args.prepared args
  have before := SuccessRoute.launchPrefix tensor device (kernel (caller:=caller) store primitives casts selection)
    args.request initial result args.prepared args route
  have hstream := WrapperStream.prefix_ready tensor device streams args.request initial args.prepared args before
  have hbound := prefix_launch_state store tensor device hs hf hd args.request initial hp args.prepared args before
  obtain ⟨hv,hi,ho⟩ := prefix_layout tensor device hl args.request initial args.prepared args before
  let ready := prefix_ready store hm tensor device hs hf hc hd hl ha args.request initial hp args.prepared args before hprofile
  have hexec := route.executed
  change ((kernel (caller:=caller) store primitives casts selection args).run current).result=.ok () at hexec
  obtain ⟨produced,hselect,hexec,hkernel⟩ := kernel_success store primitives casts selection args current ready hexec
  let c := config store primitives casts args current ready
  have hhistory := action_success_history store primitives casts args current ready produced hexec
  have hmemory := route_final_memory tensor device hmem (kernel (caller:=caller) store primitives casts selection)
    args.request initial result args.prepared args route
  change _=((kernel (caller:=caller) store primitives casts selection args).run current).state.memory at hmemory
  rw [hkernel] at hmemory
  have hfinish := route.finished
  change ((finish device args.prepared).run ((device.leave args.prepared.inputs.x.device).run
    ((kernel (caller:=caller) store primitives casts selection args).run current).state).1).result=.ok result at hfinish
  rw [hkernel] at hfinish
  have hfinal := route.finalState
  change _=((finish device args.prepared).run ((device.leave args.prepared.inputs.x.device).run
    ((kernel (caller:=caller) store primitives casts selection args).run current).state).1).state at hfinal
  rw [hkernel] at hfinal
  have hplacement : Placement store args.request args.prepared args current c := ⟨rfl,rfl,rfl,rfl,rfl⟩
  have hreturned := dispatch_suffix_returned store hm hh hr tensor device hd hmask args.request initial current
    args.prepared args hp hbound hv hi ho c hplacement (compiledBenchmarks produced) produced.selected.artifact
    template result hfinish
  have horacle := StorageMachine.dispatch_return c (compiledBenchmarks produced) produced.selected.artifact
    current.memory template hhistory
  have hresult : returnedObserve store
      ((surround tensor device (kernel (caller:=caller) store primitives casts selection) args.request).run initial).state
      args.prepared (HostReturn.seedCount (args.prepared.flags args.request) args.prepared.validated.M) result=
      produced.selected.artifact.observe c.ops c.casts preparedRegions c.strides c.M c.eps c.p
        (TypedPhysical.pull (PhysicalDomain.layout c.allocations) current.memory template) := by
    rw [hfinal]
    exact hreturned.trans horacle
  refine ⟨ready,produced,hselect,produced.selected.compiled,hstream,?_,hkernel,hhistory,hmemory,hresult,?_,?_,?_,?_⟩
  · intro row col hrow role hactive hlane hread
    obtain ⟨value,he,_⟩ := active_role_coordinate store hm tensor args.request initial current args.prepared args
      hbound hv hi ho row col hrow role hactive hlane
    exact ⟨value,he,WrapperStream.read_role_ready streams args.request args.prepared current hstream role hread value he⟩
  · exact hresult.trans (produced.selected.artifact.observe_eq c.ops c.casts preparedRegions c.strides
      c.M c.eps c.p (TypedPhysical.pull (PhysicalDomain.layout c.allocations) current.memory template))
  · intro row col hrow role hactive hlane hrole
    exact prepared_input_original store hm tensor args.request initial current args.prepared args
      hbound hv hi ho c rfl row col hrow role hactive hlane hrole
  · intro row destination
    have hstore := produced.selected.artifact.observe_store c.ops c.casts preparedRegions c.strides c.M c.eps c.p
      (TypedPhysical.pull (PhysicalDomain.layout c.allocations) current.memory template) row destination
    rw [←StorageMachine.snapshot_view c current.memory template] at hstore
    exact hstore.trans (congrArg (fun value => Compiler.storeCoordinate c.flags value row destination) hresult.symm)
  · exact (surround tensor device (kernel (caller:=caller) store primitives casts selection) args.request).frame initial

/-- Full actual successful wrapper in the source default input precision, with
all explicit output precision branches. No output dtype restriction, complete
kernel equation, positive-size condition or finite-only premise is imposed. -/
theorem surround_return
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hmem : SuffixMemory device)
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial) (hprofile : DefaultInputs q)
    (result : Returned)
    (h : ((surround tensor device (kernel (caller:=caller) store primitives casts selection) q).run initial).result=.ok result)
    (template : State α) :
    ∃ p args, SuccessRoute tensor device (kernel (caller:=caller) store primitives casts selection) q initial result p args ∧
      ReturnPost store tensor device streams primitives casts selection q initial result p args template := by
  obtain ⟨p,args,route⟩ := surround_success_route tensor device (kernel (caller:=caller) store primitives casts selection) q initial result h
  exact ⟨p,args,route,route_return store hm hh hr tensor device hs hf hc hd hl ha hmask hmem streams primitives casts selection
    q initial hp hprofile result p args route template⟩

#axiomsClean route_return
#axiomsClean surround_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
