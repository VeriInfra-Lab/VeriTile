import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageObservation
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperObservationFrame

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

/-- Actual native dispatch, device exit and returned-view creation compose to
all eight logical tensor results, including the original fallback identity. -/
theorem dispatch_suffix_returned
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hl : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hd : DispatchStorage store device) (hmask : MaskSplit store device)
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (hp : RequestPresent store q initial)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : Placement store q p args current c)
    (benchmarks : List (StorageMachine.DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (template : State α)
    (result : Returned)
    (hfinish : ((finish device p).run ((device.leave p.inputs.x.device).run
      {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}).1).result=.ok result) :
    let native := {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}
    let exited := ((device.leave p.inputs.x.device).run native).1
    returnedObserve store ((finish device p).run exited).state p
      (HostReturn.seedCount (p.flags q) p.validated.M) result=
      TypedSource.observe (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N
        (TypedPhysical.pull (PhysicalDomain.layout c.allocations) native.memory template) := by
  let native := {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}
  let exited := ((device.leave p.inputs.x.device).run native).1
  have hpresent := (prepared_present store tensor q initial current p args hp h).host_transfer
    store hh current native rfl p
  have he : store.Extends native exited := hd.leave p.inputs.x.device native
  have hreturn := finish_observe store device hd hmask p exited (hpresent.transfer he)
    (HostReturn.seedCount (p.flags q) p.validated.M) result
    (prepared_mask_shape store tensor q initial current p args h) hfinish
  exact hreturn.trans ((preparedObserve_extends store native exited he p hpresent _).trans
    (prepared_observe_native store hm hh hl tensor q initial current p args hp h hv hi ho c hc
      benchmarks selected template))

#axiomsClean dispatch_suffix_returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
