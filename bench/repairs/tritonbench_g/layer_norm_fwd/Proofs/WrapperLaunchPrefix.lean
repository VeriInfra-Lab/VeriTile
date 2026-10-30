import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLaunchAdmission

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

/-- Only the actual host prefix ending at launch admission. No kernel run or
returned output is required to construct the native configuration. -/
structure LaunchPrefix (tensor : TensorAPI world) (device : DispatchAPI world)
    (q : Request) (initial : S) (p : Prepared) (args : LaunchArguments) : Prop where
  prepared : ((prepare tensor q).run initial).result=.ok p
  entered : ((device.enter p.inputs.x.device).run (preparedState tensor q initial)).2=.ok ()
  arguments : ((Wrapper.arguments device q p).run
    (enteredState device p (preparedState tensor q initial))).result=.ok args
  admitted : ((device.admit args).run (argumentsState device q p
    (enteredState device p (preparedState tensor q initial)))).2=.ok ()

theorem SuccessRoute.launchPrefix (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (initial : S)
    (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device kernel q initial result p args) :
    LaunchPrefix tensor device q initial p args :=
  ⟨route.prepared,route.entered,route.arguments,route.admitted⟩

theorem prefix_launch_state (store : TensorStore S Descriptor Value)
    (tensor : TensorAPI world) (device : DispatchAPI world)
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor) (hd : DispatchStorage store device)
    (q : Request) (initial : S) (hp : RequestPresent store q initial) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args) :
    LaunchStateBound store tensor q initial p args (launchState tensor device q initial p args) := by
  have hprepare := prepare_state store tensor hs q initial hp p route.prepared
  have he := setup_preserves store tensor device hd q initial p args
  have hpersist := persists_prepare store tensor hs q initial
  obtain ⟨si,hsi,hinput,hbatch⟩ := prepare_storage store tensor hs hf q initial hp p route.prepared
  refine ⟨hpersist.trans he,?_,hprepare.2.1.transfer he,?_,hprepare.2.2,si,hsi,hinput,hbatch.transfer he⟩
  · exact hprepare.1.transfer hp (store.extends_refl initial) he
  · exact arguments_bound device q p _ args route.arguments

theorem prefix_layout (tensor : TensorAPI world) (device : DispatchAPI world)
    (hl : LayoutAPI tensor) (q : Request) (initial : S) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args) :
    ValidatedFacts q p.validated ∧ InputsDense p.inputs ∧ OutputsDense p.outputs :=
  ⟨prepare_validated tensor q initial p route.prepared,prepare_dense tensor hl q initial p route.prepared⟩

theorem prefix_admission (tensor : TensorAPI world) (device : DispatchAPI world)
    (ha : DispatchAdmission device) (q : Request) (initial : S) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args) (hs : SelectedStorage q p) :
    NativeABI.Accepted p.validated.M p.validated.N p.block := by
  have htype : p.inputs.x.dtype=DType.fp32 := hs .x trivial p.inputs.x rfl
  have hb := prepare_block_fp32 tensor q initial p route.prepared htype
  have hg := ha.grid args _ route.admitted
  have hpos := ha.block args _ route.admitted
  have hargs := arguments_bound device q p _ args route.arguments
  rw [hargs.prepared] at hg hpos
  exact ⟨hg,(prepare_success tensor q initial p route.prepared).1,hpos,hb⟩

#axiomsClean SuccessRoute.launchPrefix
#axiomsClean prefix_launch_state
#axiomsClean prefix_layout
#axiomsClean prefix_admission
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
