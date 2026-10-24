import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperCallerStorage
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLaunchPrefix
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperDispatchFamily

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α Host : Type} {caller : Address → Prop}

/-- Local address and layout facts for the actual arguments and predecessor
state. These contain no output values or native execution premise. -/
structure PreparedReady (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (args : LaunchArguments) (current : HostState α Host) (caller : Address → Prop) : Prop where
  accepted : NativeABI.Accepted args.prepared.validated.M args.prepared.validated.N args.prepared.block
  storage : PhysicalDomain.Domain (args.prepared.flags args.request) preparedRegions (argumentStrides args)
    args.prepared.validated.M args.prepared.validated.N (preparedAllocations store current args.prepared)
  outside : ∀ i, caller i → ∀ row : Fin args.prepared.validated.M, ∀ region offset,
    Contract.written (args.prepared.flags args.request) preparedRegions (argumentStrides args)
      args.prepared.validated.M args.prepared.validated.N row.val region offset →
    (PhysicalDomain.layout (preparedAllocations store current args.prepared)).address (region,offset)≠i.val
  geometry : NativeGeometry.Geometry (args.prepared.flags args.request) preparedRegions (argumentStrides args)
    args.prepared.validated.M args.prepared.validated.N

/-- Successful host preparation and ordinary library/runtime contracts suffice
to construct all local native launch obligations before any kernel execution. -/
theorem prefix_ready
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial)
    (hc : CallerStorage store initial caller) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args) (hprofile : SelectedStorage q p) :
    PreparedReady store args (launchState tensor device q initial p args) caller := by
  have h := prefix_launch_state store tensor device hs hf hd q initial hp p args route
  obtain ⟨hv,hi,ho⟩ := prefix_layout tensor device hl q initial p args route
  have haccepted := prefix_admission tensor device ha q initial p args route hprofile
  have hstorage := prepared_storage_domain store hm tensor q initial _ p args h hv hi ho hprofile
  have houtside := prepared_caller_outside store hm tensor q initial _ p args h hv hi ho hc
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

/-- The concrete launch uses the caller's actual argument words and tensor
pointer descriptors. Schedule choices remain compiler/runtime data. -/
def preparedLaunch
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α)
    (args : LaunchArguments) (current : HostState α Host)
    (h : PreparedReady store args current caller)
    (benchmarks : List (DispatchEvent args.prepared.block)) (selected : Schedule args.prepared.block) :
    NativeLaunch α caller args.prepared.block where
  config := configOfArguments ops encoding args selected.reductions selected.plan selected.plan1
    preparedRegions (preparedAllocations store current args.prepared) h.accepted h.storage
  benchmarks := benchmarks
  selected := selected
  fresh := h.outside
  separated := prepared_read_write_separated _
  geometry := h.geometry

theorem preparedLaunch_arguments
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ops : NativeRow.Ops α) (encoding : ScalarEncoding α)
    (args : LaunchArguments) (current : HostState α Host)
    (h : PreparedReady store args current caller)
    (benchmarks : List (DispatchEvent args.prepared.block)) (selected : Schedule args.prepared.block) :
    ArgumentsMatch encoding args (preparedLaunch store ops encoding args current h benchmarks selected).config :=
  configOfArguments_match ops encoding args selected.reductions selected.plan selected.plan1
    preparedRegions (preparedAllocations store current args.prepared) h.accepted h.storage

#axiomsClean prefix_ready
#axiomsClean preparedLaunch_arguments
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
