import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedStorage
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperRoute

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

def launchState (tensor : TensorAPI world) (device : DispatchAPI world) (q : Request)
    (s : S) (p : Prepared) (args : LaunchArguments) : S :=
  admittedState device args (argumentsState device q p
    (enteredState device p (preparedState tensor q s)))

theorem setup_preserves (store : TensorStore S Descriptor Value)
    (tensor : TensorAPI world) (device : DispatchAPI world) (hd : DispatchStorage store device)
    (q : Request) (s : S) (p : Prepared) (args : LaunchArguments) :
    store.Extends (preparedState tensor q s) (launchState tensor device q s p args) := by
  let sp := preparedState tensor q s
  let se := enteredState device p sp
  let sa := argumentsState device q p se
  exact (hd.enter p.inputs.x.device sp).trans
    ((persists_arguments store device hd q p se).trans (hd.admit args sa))

structure LaunchStateBound (store : TensorStore S Descriptor Value) (tensor : TensorAPI world)
    (q : Request) (initial : S) (p : Prepared) (args : LaunchArguments) (current : S) : Prop where
  preserved : store.Extends initial current
  inputs : InputsMaterialized store q initial p.inputs current
  outputs : OutputsAllocated store tensor q p.validated p.inputs p.outputs current
  arguments : ArgumentsBound args p q
  width : p.validated.N≤p.block
  storage : ∃ allocationStart, store.Extends initial allocationStart ∧
    InputsMaterialized store q initial p.inputs allocationStart ∧
    AllocationBatch store allocationStart (outputTensors p.outputs) current

/-- All prepared tensor contents, descriptors and physical allocation
separation reach the exact state on which the kernel action is invoked. The
success of each prefix call is extracted from the actual wrapper result. -/
theorem route_launch_state (store : TensorStore S Descriptor Value)
    (tensor : TensorAPI world) (device : DispatchAPI world)
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor) (hd : DispatchStorage store device)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (s : S)
    (hp : RequestPresent store q s) (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device kernel q s result p args) :
    LaunchStateBound store tensor q s p args (launchState tensor device q s p args) := by
  have hprepare := prepare_state store tensor hs q s hp p route.prepared
  have he := setup_preserves store tensor device hd q s p args
  have hpersist := persists_prepare store tensor hs q s
  obtain ⟨si,hsi,hinput,hbatch⟩ := prepare_storage store tensor hs hf q s hp p route.prepared
  refine ⟨hpersist.trans he,?_,hprepare.2.1.transfer he,?_,hprepare.2.2,si,hsi,hinput,hbatch.transfer he⟩
  · exact hprepare.1.transfer hp (store.extends_refl s) he
  · exact arguments_bound device q p _ args route.arguments

theorem surround_launch_state (store : TensorStore S Descriptor Value)
    (tensor : TensorAPI world) (device : DispatchAPI world)
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor) (hd : DispatchStorage store device)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (s : S)
    (hp : RequestPresent store q s) (result : Returned)
    (h : ((surround tensor device kernel q).run s).result=.ok result) :
    ∃ p args, SuccessRoute tensor device kernel q s result p args ∧
      LaunchStateBound store tensor q s p args (launchState tensor device q s p args) := by
  obtain ⟨p,args,route⟩ := surround_success_route tensor device kernel q s result h
  exact ⟨p,args,route,route_launch_state store tensor device hs hf hd kernel q s hp result p args route⟩

#axiomsClean setup_preserves
#axiomsClean route_launch_state
#axiomsClean surround_launch_state
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
