import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageAddress
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperCallerReady
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperSelectedRequest

/-! Default FP32 floating inputs, with every source-explicit output dtype
retained. DTypes remain actual API descriptors rather than an enumeration of
unrequested input precision combinations. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

def DefaultInputs (q : Request) : Prop :=
  ∀ role, sourceInputRole role → ∀ tensor,
    originalRoleTensor q role=some tensor → tensor.dtype=DType.fp32

def requestedResidualDType (q : Request) : DType :=
  if q.residual.isSome then DType.fp32 else q.residualDtype.getD DType.fp32

def requestedRoleDType (q : Request) : LayerNormFwd.Address.Role → DType
  | .y | .y1 => q.outDtype.getD DType.fp32
  | .residualOut => requestedResidualDType q
  | .seeds | .seeds1 => .int64
  | .mask | .mask1 => .bool
  | _ => .fp32

/-- The residual input overrides residual_dtype, exactly as the wrapper does.
Output and residual conversions do not change the FP32 accumulation profile. -/
theorem requested_role_dtype
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hp : DefaultInputs q) (role : LayerNormFwd.Address.Role) (value : Tensor)
    (he : roleTensor p role=some value) : value.dtype=requestedRoleDType q role := by
  have hx : p.inputs.x.dtype=DType.fp32 := h.inputs.x.dtype.trans (hp .x trivial q.x rfl)
  have hy : p.outputs.y.dtype=q.outDtype.getD DType.fp32 := by
    simpa only [outputRequests,hx] using h.outputs.1.dtype
  have hr : p.validated.residualDtype.getD DType.fp32=requestedResidualDType q := by
    rw [hv.residualDtype]
    cases hh : q.residual with
    | none => simp [requestedResidualDType,hh]
    | some value => simpa [requestedResidualDType,hh] using hp .residual trivial value hh
  by_cases hs : sourceInputRole role
  · obtain ⟨source,hsource,hmaterialized⟩ := role_materialized store tensor q initial current p args h role hs value he
    have hd := hmaterialized.dtype.trans (hp role hs source hsource)
    cases role <;> simp_all [sourceInputRole,requestedRoleDType]
  · cases role with
    | x | weight | residual | x1 | bias | weight1 | bias1 | rowscale | rowscale1 =>
      exact False.elim (hs True.intro)
    | y => exact Option.some.inj he ▸ hy
    | y1 =>
      simpa only [outputRequests,requestedRoleDType,hy] using (allocated_of_some h.outputs.2.1 value he).dtype
    | residualOut =>
      simpa only [outputRequests,requestedRoleDType,hx,hr] using (allocated_of_some h.outputs.2.2.1 value he).dtype
    | rstd => exact Option.some.inj he ▸ h.outputs.2.2.2.2.1.dtype
    | mean => exact (allocated_of_some h.outputs.2.2.2.1 value he).dtype
    | seeds => exact (allocated_of_some h.outputs.2.2.2.2.2.1 value he).dtype
    | seeds1 => exact (allocated_of_some h.outputs.2.2.2.2.2.1 value he).dtype
    | mask => exact (allocated_of_some h.outputs.2.2.2.2.2.2 value he).dtype
    | mask1 => exact (allocated_of_some h.outputs.2.2.2.2.2.2 value he).dtype

theorem requested_role_width
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hp : DefaultInputs q) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role) :
    (preparedAllocations store current p (LayerNormFwd.Address.region preparedRegions role)).width=
      (requestedRoleDType q role).bytes := by
  obtain ⟨value,he⟩ := active_role_tensor store tensor q initial current p args h role ha
  rw [preparedAllocation_role store current p role value he,
    hm.width current value (prepared_tensor_present store tensor q initial current p args h role value he),
    requested_role_dtype store tensor q initial current p args h hv hp role value he]

/-- Width-independent domain derived from the same actual host allocations.
This admits all explicit output dtype descriptors accepted by the tensor API. -/
theorem prepared_requested_domain
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) :
    StorageMachine.Domain (p.flags q) preparedRegions (argumentStrides args)
      p.validated.M p.validated.N (preparedAllocations store current p) :=
  ⟨launch_argument_metadata store hm tensor q initial current p args h hv,
    prepared_row_bounds store hm tensor q initial current p args h hv hi ho,
    prepared_fresh_writes store hm tensor q initial current p args h hv hi ho⟩

/-- Admission uses the source input element size. Explicit output types do
not enter the 65536-byte block-width limit. No positive-N condition is added. -/
theorem prefix_requested_admission
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (ha : DispatchAdmission device) (q : Request) (initial : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args)
    (h : LaunchStateBound store tensor q initial p args (launchState tensor device q initial p args))
    (hp : DefaultInputs q) : NativeABI.Accepted p.validated.M p.validated.N p.block := by
  have hx := h.inputs.x.dtype.trans (hp .x trivial q.x rfl)
  have hb := prepare_block_fp32 tensor q initial p route.prepared hx
  have hg := ha.grid args _ route.admitted
  have hpos := ha.block args _ route.admitted
  rw [h.arguments.prepared] at hg hpos
  exact ⟨hg,(prepare_success tensor q initial p route.prepared).1,hpos,hb⟩

#axiomsClean requested_role_dtype
#axiomsClean requested_role_width
#axiomsClean prepared_requested_domain
#axiomsClean prefix_requested_admission
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
