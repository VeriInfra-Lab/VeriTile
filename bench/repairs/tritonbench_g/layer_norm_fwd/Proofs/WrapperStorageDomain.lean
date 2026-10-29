import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPhysicalFresh

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 4000000

def roleDType : LayerNormFwd.Address.Role → DType
  | .seeds | .seeds1 => .int64
  | .mask | .mask1 => .bool
  | _ => .fp32

/-- The currently proved native storage profile, expressed on actual tensor
metadata. This is not a finite-value or accuracy hypothesis. Other floating
storage profiles remain separate unfinished program work. -/
def SelectedStorage (q : Request) (p : Prepared) : Prop :=
  ∀ role, LayerNormFwd.Address.active (p.flags q) role → ∀ tensor,
    roleTensor p role=some tensor → tensor.dtype=roleDType role

theorem role_dtype_width (role : LayerNormFwd.Address.Role) :
    (roleDType role).bytes=PhysicalDomain.roleWidth role := by cases role <;> rfl

theorem active_role_tensor {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role) : ∃ value, roleTensor p role=some value := by
  cases role with
  | x => exact ⟨p.inputs.x,rfl⟩
  | y => exact ⟨p.outputs.y,rfl⟩
  | weight => exact ⟨p.inputs.weight,rfl⟩
  | rstd => exact ⟨p.outputs.rstd,rfl⟩
  | residual => exact tensor_of_isSome _ ha
  | x1 => exact tensor_of_isSome _ ha
  | bias => exact tensor_of_isSome _ ha
  | weight1 => exact tensor_of_isSome _ ha
  | bias1 => exact tensor_of_isSome _ ha.2
  | rowscale => exact tensor_of_isSome _ ha
  | rowscale1 => exact tensor_of_isSome _ ha.2
  | residualOut => exact tensor_of_isSome _ ha
  | mask => exact tensor_of_isSome _ ha.2
  | mask1 => exact tensor_of_isSome _ ha.2.2
  | y1 =>
    obtain ⟨value,he,_⟩ := allocated_of_enabled h.outputs.2.1 ha
    exact ⟨value,he⟩
  | mean =>
    have henabled : (!q.rms)=Bool.true := by simpa [LayerNormFwd.Address.active,Prepared.flags] using ha
    obtain ⟨value,he,_⟩ := allocated_of_enabled h.outputs.2.2.2.1 henabled
    exact ⟨value,he⟩
  | seeds =>
    obtain ⟨value,he,_⟩ := allocated_of_enabled h.outputs.2.2.2.2.2.1 ha
    exact ⟨value,he⟩
  | seeds1 =>
    obtain ⟨value,he,_⟩ := allocated_of_enabled h.outputs.2.2.2.2.2.1 ha.2
    exact ⟨value,he⟩

theorem prepared_tensor_present {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current)
    (role : LayerNormFwd.Address.Role) (value : Tensor) (he : roleTensor p role=some value) :
    store.present current value := by
  obtain ⟨allocationStart,_,hinputs,hbatch⟩ := h.storage
  have hregion : regionTensor p (LayerNormFwd.Address.region preparedRegions role)=some value := by
    rw [regionTensor_role,he]
  rcases regionTensor_member p _ value hregion with hi | ho
  · exact (hbatch.preserved value (inputViews_present hinputs _ value hi)).1
  · apply hbatch.present value
    have hmap := List.mem_map_of_mem (f:=Prod.snd) ho
    simpa only [outputViews_values] using hmap

theorem prepared_role_widths {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hs : SelectedStorage q p)
    (role : LayerNormFwd.Address.Role) (ha : LayerNormFwd.Address.active (p.flags q) role) :
    (preparedAllocations store current p (LayerNormFwd.Address.region preparedRegions role)).width=
      PhysicalDomain.roleWidth role := by
  obtain ⟨value,he⟩ := active_role_tensor store tensor q initial current p args h role ha
  rw [preparedAllocation_role store current p role value he,
    hm.width current value (prepared_tensor_present store tensor q initial current p args h role value he),
    hs role ha value he,role_dtype_width]

/-- Complete native storage domain constructed from actual prepared tensor
descriptors. Metadata, typed widths, all active spans and fresh writes are
derived independently of kernel execution or output values. -/
theorem prepared_storage_domain {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) (hs : SelectedStorage q p) :
    PhysicalDomain.Domain (p.flags q) preparedRegions (argumentStrides args)
      p.validated.M p.validated.N (preparedAllocations store current p) :=
  ⟨launch_argument_metadata store hm tensor q initial current p args h hv,
    prepared_role_widths store hm tensor q initial current p args h hs,
    prepared_row_bounds store hm tensor q initial current p args h hv hi ho,
    prepared_fresh_writes store hm tensor q initial current p args h hv hi ho⟩

#axiomsClean role_dtype_width
#axiomsClean active_role_tensor
#axiomsClean prepared_tensor_present
#axiomsClean prepared_role_widths
#axiomsClean prepared_storage_domain
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
