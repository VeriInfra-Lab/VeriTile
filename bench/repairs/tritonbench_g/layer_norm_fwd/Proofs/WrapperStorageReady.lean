import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageProvenance
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageExecution

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

structure PreparedReady (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (args : LaunchArguments) (current : HostState α Host) (caller : Address → Prop) : Prop where
  accepted : NativeABI.Accepted args.prepared.validated.M args.prepared.validated.N args.prepared.block
  storage : StorageMachine.Domain (args.prepared.flags args.request) preparedRegions (argumentStrides args)
    args.prepared.validated.M args.prepared.validated.N (preparedAllocations store current args.prepared)
  outside : ∀ i, caller i → ∀ k,
    AliasMemory.Writes (args.prepared.flags args.request) preparedRegions (argumentStrides args)
      args.prepared.validated.M args.prepared.validated.N k →
    (PhysicalDomain.layout (preparedAllocations store current args.prepared)).address k≠i.val
  geometry : NativeGeometry.Geometry (args.prepared.flags args.request) preparedRegions (argumentStrides args)
    args.prepared.validated.M args.prepared.validated.N
  requested : StorageCompiler.RequestedStorage args

/-- Before any benchmark, the actual prefix supplies the entire address,
freshness and requested-precision domain. Explicit output dtypes are unrestricted
by this theorem; their ordinary allocation/compiler outcomes remain observable. -/
theorem prefix_ready
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial)
    (p : Prepared) (args : LaunchArguments) (route : LaunchPrefix tensor device q initial p args)
    (hprofile : DefaultInputs q) : PreparedReady store args (launchState tensor device q initial p args) caller := by
  have h := prefix_launch_state store tensor device hs hf hd q initial hp p args route
  obtain ⟨hv,hi,ho⟩ := prefix_layout tensor device hl q initial p args route
  obtain ⟨allocationStart,hbatch⟩ := prefix_caller_storage store tensor device hs hc hd q initial p args route
  have haccepted := prefix_requested_admission store tensor device ha q initial p args route h hprofile
  have hstorage := prepared_requested_domain store hm tensor q initial _ p args h hv hi ho
  have houtside := prepared_full_caller_outside store hm tensor q initial _ allocationStart p args h hv hi ho hbatch
  have hgeometry := launch_geometry store hm tensor q initial _ p args h hv ho preparedRegions prepared_output_names
  refine ⟨?_,?_,?_,?_,?_⟩
  · simpa only [h.arguments.prepared] using haccepted
  · simpa only [h.arguments.prepared,h.arguments.request] using hstorage
  · intro i hi k hk
    obtain ⟨row,hr,hw⟩ := hk
    have hrow : row<p.validated.M := by simpa only [h.arguments.prepared] using hr
    have hw' : Contract.written (p.flags q) preparedRegions (argumentStrides args)
        p.validated.M p.validated.N row k.1 k.2 := by
      simpa only [h.arguments.prepared,h.arguments.request] using hw
    simpa only [h.arguments.prepared] using houtside i hi ⟨row,hrow⟩ k.1 k.2 hw'
  · simpa only [h.arguments.prepared,h.arguments.request] using hgeometry
  · intro role value he
    have he' : roleTensor p role=some value := by simpa only [h.arguments.prepared] using he
    simpa only [h.arguments.request] using requested_role_dtype store tensor q initial _ p args h hv hprofile role value he'

def config (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (args : LaunchArguments) (current : HostState α Host) (h : PreparedReady store args current caller) :
    StorageMachine.Config α args.prepared.block where
  ops := primitives.ops
  casts := casts
  profile := StorageCompiler.Profile.ofRequest args.request
  flags := args.prepared.flags args.request
  strides := argumentStrides args
  M := args.prepared.validated.M
  N := args.prepared.validated.N
  eps := primitives.encoding.encode args.request.eps
  p := primitives.encoding.encode args.request.dropoutP
  allocations := preparedAllocations store current args.prepared
  accepted := h.accepted
  storage := h.storage
  geometry := h.geometry

#axiomsClean prefix_ready
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
