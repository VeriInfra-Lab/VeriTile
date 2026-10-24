import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedLaunch
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPhysicalView

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

/-- Logical reads at actual prepared coordinates equal the corresponding
native memory cells. The post-kernel state may have arbitrary cell values;
only its ordinary host descriptor metadata must be unchanged. -/
theorem prepared_role_readback
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current native : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : Config α block) (hc : c.allocations=preparedAllocations store current p)
    (hn : native.host=current.host) (template : State α)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role)
    (hl : NativeAccess.laneActive p.validated.N col role) :
    ∃ value, roleTensor p role=some value ∧ store.present native value ∧
      store.read native value (roleIndex p.validated.M row col role)=
        (view c native.memory template).mem (LayerNormFwd.Address.region preparedRegions role)
          (LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role) := by
  obtain ⟨value,he,hp,hpositive,hcoordinate,hoffset⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  have hp' := hh.present current native hn.symm value hp
  refine ⟨value,he,hp',?_⟩
  apply Eq.symm
  apply descriptor_read store hm c native value hp' hpositive _ _ _ hcoordinate _ hoffset template
  rw [hc,preparedAllocation_role store current p role value he]
  exact hh.descriptor current native hn.symm value

/-- Exit and view creation transport the actual prepared readback to the
wrapper's final state by ordinary tensor persistence. -/
theorem prepared_role_readback_suffix
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current native final : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : Config α block) (hc : c.allocations=preparedAllocations store current p)
    (hn : native.host=current.host) (hext : store.Extends native final) (template : State α)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role)
    (hl : NativeAccess.laneActive p.validated.N col role) :
    ∃ value, roleTensor p role=some value ∧ store.present final value ∧
      store.read final value (roleIndex p.validated.M row col role)=
        (view c native.memory template).mem (LayerNormFwd.Address.region preparedRegions role)
          (LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role) := by
  obtain ⟨value,he,hp,hread⟩ := prepared_role_readback store hm hh tensor q initial current native
    p args h hv hi ho c hc hn template row col hr role ha hl
  exact ⟨value,he,(hext value hp).1,((hext value hp).2.2 _).trans hread⟩

def originalRoleTensor (q : Request) : LayerNormFwd.Address.Role → Option Tensor
  | .x => some q.x | .weight => some q.weight
  | .residual => q.residual | .x1 => q.x1
  | .bias => q.bias | .weight1 => q.weight1 | .bias1 => q.bias1
  | .rowscale | .rowscale1 => q.rowscale
  | _ => none

def sourceInputRole : LayerNormFwd.Address.Role → Prop
  | .x | .weight | .residual | .x1 | .bias | .weight1 | .bias1 | .rowscale | .rowscale1 => True
  | _ => False

theorem role_materialized
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current)
    (role : LayerNormFwd.Address.Role) (hi : sourceInputRole role)
    (value : Tensor) (he : roleTensor p role=some value) :
    ∃ source, originalRoleTensor q role=some source ∧ Materialized store source initial value current := by
  cases role <;> simp only [sourceInputRole] at hi
  all_goals try contradiction
  case x =>
    have hv : p.inputs.x=value := Option.some.inj he
    exact ⟨q.x,rfl,hv ▸ h.inputs.x⟩
  case weight =>
    have hv : p.inputs.weight=value := Option.some.inj he
    exact ⟨q.weight,rfl,hv ▸ h.inputs.weight⟩
  case residual => exact materialized_of_some h.inputs.residual value he
  case x1 => exact materialized_of_some h.inputs.x1 value he
  case bias => exact materialized_of_some h.inputs.bias value he
  case weight1 => exact materialized_of_some h.inputs.weight1 value he
  case bias1 => exact materialized_of_some h.inputs.bias1 value he
  case rowscale => exact materialized_of_some h.inputs.rowscale value he
  case rowscale1 => exact materialized_of_some h.inputs.rowscale value he

/-- Every native source-input operand refers to the original logical input
coordinate, including inputs carrying Torch's lazy negative bit. -/
theorem prepared_input_original
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : Config α block) (hc : c.allocations=preparedAllocations store current p) (template : State α)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role)
    (hl : NativeAccess.laneActive p.validated.N col role) (hrole : sourceInputRole role) :
    ∃ source, originalRoleTensor q role=some source ∧
      (view c current.memory template).mem (LayerNormFwd.Address.region preparedRegions role)
        (LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role)=
      store.read initial source (roleIndex p.validated.M row col role) := by
  obtain ⟨value,he,_,_,hcoordinate,hoffset⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  obtain ⟨source,hsource,hmaterialized⟩ := role_materialized store tensor q initial current p args h role hrole value he
  refine ⟨source,hsource,?_⟩
  apply descriptor_materialized_read store hm c initial current source value hmaterialized _ _ _ hcoordinate _ hoffset template
  rw [hc]
  exact preparedAllocation_role store current p role value he

#axiomsClean prepared_role_readback
#axiomsClean prepared_role_readback_suffix
#axiomsClean role_materialized
#axiomsClean prepared_input_original
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
