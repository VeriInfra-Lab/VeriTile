import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.RealReturn
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageReturn

/-! Whole successful production wrapper, interpreted at arbitrary original Real
physical floating cells and independent Real scalar positions. The actual host
branches, output precision layouts, generated discrete seeds and returned views
belong to the same call. No equality between FP results and Real values is used.
The historical FP proof and its local numerical obligations remain unchanged. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision.Correctness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option Elab.async false
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

/-- All observations use actual returned tensor descriptors. The formula reads
only the arbitrary pre-benchmark input snapshot; its original-request mapping
and both physical frames are conclusions of the same-call theorem. -/
def RealPost
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (ρ : HostState α Host → Address → ℝ) (epsReal dropoutReal : ℝ)
    (q : Request) (initial : HostState α Host) (result : Returned) (p : Prepared)
    (args : LaunchArguments) (template : BlockState) : Prop :=
  let current := launchState tensor device q initial p args
  let layout := PhysicalDomain.layout (preparedAllocations store current p)
  let f := p.flags q
  let finalHost := ((surround tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection) q).run initial).state
  ∃ (produced : StorageCompiler.ProducedDispatch selection.compiler args current.host)
    (hBlock : Random.ArangeSizeDomain p.block) (benchmarkMemory : RealMemory) (final : BlockState) (physical : RealMemory),
    selection.select args current.host=.ok produced ∧
    RealHistory.Replay f (argumentStrides args) p.validated.M p.validated.N epsReal dropoutReal p.block hBlock layout template
      (RealHistory.eraseEvents (WrapperStorage.compiledBenchmarks produced)) (interpret ρ current) benchmarkMemory ∧
    RealHistory.InputEqual layout (interpret ρ current) benchmarkMemory ∧
    physical=AliasMemory.runPhysical layout benchmarkMemory
      (AliasGrid.events f preparedRegions (argumentStrides args) p.validated.M p.validated.N final) ∧
    LaunchStateBound store tensor q initial p args current ∧
    InputsMaterialized (shadowStore store ρ) q initial p.inputs current ∧
    Nonempty (Kernel.GridLaunchedOrdinary
      ((repaired f preparedRegions (argumentStrides args) p.validated.M p.validated.N
        epsReal dropoutReal p.block hBlock).eraseDType.toAlgKernel)
      (WholeGrid.grid p.validated.M) (AliasMemory.pull layout benchmarkMemory template) final) ∧
    AliasMemory.pull layout physical final=final ∧
    WholeGrid.Post f preparedRegions (argumentStrides args) p.validated.M p.validated.N
      epsReal dropoutReal (AliasMemory.pull layout benchmarkMemory template) final ∧
    returnedObservation store finalHost physical p (HostReturn.seedCount f p.validated.M) result=
      HostReturn.oracle f preparedRegions (argumentStrides args) p.validated.M p.validated.N
        epsReal dropoutReal (AliasMemory.pull layout (interpret ρ current) template) ∧
    (∀ row col, row<p.validated.M → ∀ role : LayerNormFwd.Address.Role,
      LayerNormFwd.Address.active f role → NativeAccess.laneActive p.validated.N col role → sourceInputRole role →
      ∃ source, originalRoleTensor q role=some source ∧
        AliasMemory.read layout (interpret ρ current)
          (LayerNormFwd.Address.region preparedRegions role,
            LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role)=
          (shadowStore store ρ).read initial source (roleIndex p.validated.M row col role)) ∧
    (∀ i : Address, (∀ k, AliasMemory.Writes f preparedRegions (argumentStrides args) p.validated.M p.validated.N k →
      layout.address k≠i.val) → physical i=interpret ρ current i) ∧
    (∀ i, caller i → physical i=interpret ρ initial i) ∧
    (∀ i, caller i → finalHost.memory i=initial.memory i) ∧
    result.y=p.outputs.y ∧ result.y1=p.outputs.y1 ∧ result.mean=p.outputs.mean ∧ result.rstd=p.outputs.rstd ∧
    result.residual=p.outputs.residualOut.getD p.inputs.originalX ∧ result.seeds=p.outputs.seeds


theorem route_real
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hsplit : SplitCoordinates store device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (ρ : HostState α Host → Address → ℝ) (laws : Transport store ρ tensor)
    (shadowStorage : StorageAPI (shadowStore store ρ) (revalueTensor store ρ tensor laws))
    (shadowDispatch : DispatchStorage (shadowStore store ρ) (revalueDispatch device))
    (tf : TensorFrames ρ tensor) (df : DispatchFrames ρ device)
    (epsReal dropoutReal : ℝ) (q : Request) (initial : HostState α Host)
    (hp : RequestPresent store q initial) (hprofile : DefaultInputs q)
    (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection)
      q initial result p args) (template : BlockState) :
    RealPost store tensor device primitives casts selection ρ epsReal dropoutReal q initial result p args template := by
  have hargs := arguments_bound device q p _ args route.arguments
  have hprepared := hargs.prepared
  have hrequest := hargs.request
  subst p
  subst q
  let current := launchState tensor device args.request initial args.prepared args
  let f := args.prepared.flags args.request
  let layout := PhysicalDomain.layout (preparedAllocations store current args.prepared)
  have before := SuccessRoute.launchPrefix tensor device
    (WrapperStorage.kernel (caller:=caller) store primitives casts selection) args.request initial result args.prepared args route
  have hbound := prefix_launch_state store tensor device hs hf hd args.request initial hp args.prepared args before
  obtain ⟨hv,hi,ho⟩ := prefix_layout tensor device hl args.request initial args.prepared args before
  have hshadow := shadow_prefix store ρ tensor laws device shadowStorage hf shadowDispatch args.request initial hp args.prepared args before
  let ready := WrapperStorage.prefix_ready store hm tensor device hs hf hc hd hl ha args.request initial hp args.prepared args before hprofile
  let c := WrapperStorage.config store primitives casts args current ready
  have placement : WrapperStorage.Placement store args.request args.prepared args current c := ⟨rfl,rfl,rfl,rfl,rfl⟩
  have htype := hbound.inputs.x.dtype.trans (hprofile .x trivial args.request.x rfl)
  have hpreparedRun : ((prepare tensor args.request).run initial).result=.ok args.prepared := route.prepared
  have hpositive : 0<args.prepared.block := ready.accepted.blockPositive
  have hBlock : Random.ArangeSizeDomain args.prepared.block := prepare_arange (w:=world (α:=α) (Host:=Host) caller) tensor args.request initial args.prepared hpreparedRun htype hpositive
  have hsep := prepared_read_write_separated f
  have hexec := route.executed
  change ((WrapperStorage.kernel (caller:=caller) store primitives casts selection args).run current).result=.ok () at hexec
  obtain ⟨produced,hselect,haction,hkernel⟩ := WrapperStorage.kernel_success store primitives casts selection args current ready hexec
  have hsuccess := WrapperStorage.action_success_history store primitives casts args current ready produced haction
  obtain ⟨benchmarkMemory,hreplay,hbenchInput,hbenchFrame⟩ := RealHistory.replay_exists c
    (WrapperStorage.compiledBenchmarks produced) current.memory hsuccess epsReal dropoutReal hBlock template (interpret ρ current)
  obtain ⟨physical,selectedStep,hselectedInput,hselectedFrame⟩ := RealHistory.launch_exists
    f (argumentStrides args) args.prepared.validated.M args.prepared.validated.N epsReal dropoutReal args.prepared.block hBlock layout template
      ready.accepted.width ready.geometry ready.storage.fresh benchmarkMemory
  obtain ⟨final,hrun,hscatter,hpost⟩ := selectedStep.witness
  have hpull : AliasMemory.pull layout physical final=final := by
    rw [hscatter]
    exact AliasGrid.source_trace_pull layout benchmarkMemory template final f preparedRegions (argumentStrides args)
      args.prepared.validated.M args.prepared.validated.N epsReal dropoutReal ready.storage.fresh hpost
  have hframe : RealHistory.Frame f (argumentStrides args) args.prepared.validated.M args.prepared.validated.N layout (interpret ρ current) physical :=
    fun i outside => (hselectedFrame i outside).trans (hbenchFrame i outside)
  have hinput : RealHistory.InputEqual layout (interpret ρ current) physical :=
    fun region hregion offset => (hselectedInput region hregion offset).trans (hbenchInput region hregion offset)
  have hmath : HostReturn.observe f preparedRegions (argumentStrides args) args.prepared.validated.M args.prepared.validated.N
      (AliasMemory.pull layout physical final)=
      HostReturn.oracle f preparedRegions (argumentStrides args) args.prepared.validated.M args.prepared.validated.N epsReal dropoutReal
        (AliasMemory.pull layout (interpret ρ current) template) := by
    rw [hpull]
    exact (ActiveGrid.observe_eq_oracle f preparedRegions (argumentStrides args) args.prepared.validated.M args.prepared.validated.N
      epsReal dropoutReal _ _ ready.geometry hsep hpost).trans
        (RealHistory.oracle_eq layout (interpret ρ current) benchmarkMemory hbenchInput f (argumentStrides args)
          args.prepared.validated.M args.prepared.validated.N epsReal dropoutReal template)
  have hfall (i : Fin args.prepared.validated.M) (j : Fin args.prepared.validated.N) :
      logicalRead store current physical args.prepared.inputs.originalX [i.val,j.val]=
        AliasMemory.read layout physical (preparedRegions.x,i.val*(argumentStrides args).x+j.val) := by
    have hindex := coordinate_matrix args.request.x args.prepared.validated.M args.prepared.validated.N i.val j.val hv.x.1 i.isLt j.isLt
    have hprior := prior_real_read store hm tensor args.request initial current args.prepared args hbound hv hi ho c placement
      (interpret ρ current) physical hframe args.request.x hp.x [i.val,j.val] hindex
    obtain ⟨source,hsource,hvalue⟩ := input_original store hm ρ tensor args.request initial current args.prepared args hbound hv hi ho
      hshadow.inputs i.val j.val i.isLt .x trivial j.isLt trivial
    have he : source=args.request.x := (Option.some.inj hsource).symm
    subst source
    have hread := hinput preparedRegions.x (by simp [StorageMachine.inputRegion]) (i.val*(argumentStrides args).x+j.val)
    rw [hbound.inputs.original]
    exact hprior.trans (((hshadow.preserved args.request.x hp.x).2.2 _).trans (hvalue.symm.trans hread.symm))
  have hobs := prepared_observation store hm tensor args.request initial current args.prepared args hbound hv hi ho physical final hfall
  let native := ((WrapperStorage.kernel (caller:=caller) store primitives casts selection args).run current).state
  let exited := ((device.leave args.prepared.inputs.x.device).run native).1
  have hhost : current.host=native.host := by
    dsimp only [native]
    rw [hkernel]
    rfl
  have hpresent := prepared_present store tensor args.request initial current args.prepared args hp hbound
  have hnative := hpresent.host_transfer store hh current native hhost args.prepared
  have hext := hd.leave args.prepared.inputs.x.device native
  have hfinish := finish_observation store physical device hd hmask hsplit args.prepared exited (hnative.transfer hext)
    (HostReturn.seedCount f args.prepared.validated.M) result (prepared_mask_shape store tensor args.request initial current args.prepared args hbound) route.finished
  have htransport := (readStore_host store hh physical current native hhost).trans
    (readStore_extends store physical native exited hext)
  have hreturned : returnedObservation store
      ((surround tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection) args.request).run initial).state
      physical args.prepared (HostReturn.seedCount f args.prepared.validated.M) result=
      HostReturn.oracle f preparedRegions (argumentStrides args) args.prepared.validated.M args.prepared.validated.N epsReal dropoutReal
        (AliasMemory.pull layout (interpret ρ current) template) := by
    rw [route.finalState]
    exact hfinish.trans ((prepared_observation_extends store physical current exited htransport args.prepared hpresent _).trans
      (hobs.trans hmath))
  refine ⟨produced,hBlock,benchmarkMemory,final,physical,hselect,hreplay,hbenchInput,hscatter,
    hbound,hshadow.inputs,hrun,hpull,hpost,hreturned,?_,hframe,?_,?_,?_⟩
  · intro row col hr role hactive hlane hrole
    exact input_original store hm ρ tensor args.request initial current args.prepared args hbound hv hi ho hshadow.inputs
      row col hr role hactive hlane hrole
  · intro i hi
    exact (hframe i (ready.outside i hi)).trans
      (prefix_caller_frame store ρ tensor laws tf device df args.request initial args.prepared args i hi)
  · exact (surround tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection) args.request).frame initial
  · exact finish_return_identity device args.prepared exited result route.finished

/-- Public same-original-request entry. Both the existing complete production
precision result and the independent arbitrary-Real wrapper result are bound
to one observed successful call. All new premises are local library transport,
address and caller-frame laws; neither a full output equation nor a second
successful Real wrapper is assumed. -/
specification same_call_real_return
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hmem : SuffixMemory device) (hsplit : SplitCoordinates store device)
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (ρ : HostState α Host → Address → ℝ) (laws : Transport store ρ tensor)
    (shadowStorage : StorageAPI (shadowStore store ρ) (revalueTensor store ρ tensor laws))
    (shadowDispatch : DispatchStorage (shadowStore store ρ) (revalueDispatch device))
    (tf : TensorFrames ρ tensor) (df : DispatchFrames ρ device)
    (epsReal dropoutReal : ℝ) (q : Request) (initial : HostState α Host)
    (hp : RequestPresent store q initial) (hprofile : DefaultInputs q) (result : Returned)
    (success : ((surround tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection) q).run initial).result=.ok result)
    (nativeTemplate : State α) (realTemplate : BlockState) :
    ∃ p args, SuccessRoute tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection)
        q initial result p args ∧
      WrapperStorage.ReturnPost store tensor device streams primitives casts selection q initial result p args nativeTemplate ∧
      RealPost store tensor device primitives casts selection ρ epsReal dropoutReal q initial result p args realTemplate := by
  obtain ⟨p,args,route⟩ := surround_success_route tensor device
    (WrapperStorage.kernel (caller:=caller) store primitives casts selection) q initial result success
  exact ⟨p,args,route,
    WrapperStorage.route_return store hm hh hr tensor device hs hf hc hd hl ha hmask hmem streams primitives casts selection
      q initial hp hprofile result p args route nativeTemplate,
    route_real store hm hh tensor device hs hf hc hd hl ha hmask hsplit primitives casts selection
      ρ laws shadowStorage shadowDispatch tf df epsReal dropoutReal q initial hp hprofile result p args route realTemplate⟩

#axiomsClean route_real
#axiomsClean same_call_real_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision.Correctness
