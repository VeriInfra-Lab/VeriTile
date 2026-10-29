import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLaunchGeometry

/-! Each active native pointer role is bound to the actual prepared tensor
and a valid logical coordinate. The source's element offset is derived from
the local layout contracts; no complete address-map premise is assumed. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def roleTensor (p : Prepared) : LayerNormFwd.Address.Role → Option Tensor
  | .x => some p.inputs.x | .weight => some p.inputs.weight
  | .residual => p.inputs.residual | .x1 => p.inputs.x1
  | .bias => p.inputs.bias | .weight1 => p.inputs.weight1 | .bias1 => p.inputs.bias1
  | .rowscale | .rowscale1 => p.inputs.rowscale
  | .y => some p.outputs.y | .y1 => p.outputs.y1 | .residualOut => p.outputs.residualOut
  | .mean => p.outputs.mean | .rstd => some p.outputs.rstd
  | .seeds | .seeds1 => p.outputs.seeds | .mask | .mask1 => p.outputs.mask

def roleIndex (M row col : Nat) : LayerNormFwd.Address.Role → List Nat
  | .x | .y | .residual | .residualOut | .x1 | .y1 | .mask => [row,col]
  | .weight | .bias | .weight1 | .bias1 => [col]
  | .mean | .rstd | .rowscale | .seeds => [row]
  | .rowscale1 | .seeds1 => [M+row]
  | .mask1 => [M+row,col]

def RoleCoordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (s : HostState α Host)
    (p : Prepared) (args : LaunchArguments) (row col : Nat) (role : LayerNormFwd.Address.Role) : Prop :=
  ∃ tensor, roleTensor p role=some tensor ∧ store.present s tensor ∧ tensor.negative=Bool.false ∧
    Coordinate tensor (roleIndex p.validated.M row col role) ∧
    LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role=
      coordinateOffset tensor.strides (roleIndex p.validated.M row col role)

theorem materialized_of_some {S Value : Type} {store : TensorStore S Descriptor Value}
    {original output : Option Tensor} {initial current : S}
    (h : OptionMaterialized store original initial output current) (tensor : Tensor) (he : output=some tensor) :
    ∃ source, original=some source ∧ Materialized store source initial tensor current := by
  rw [he] at h
  cases original with
  | none => exact False.elim h
  | some source => exact ⟨source,rfl,h⟩

theorem optional_materialized_isSome {S Value : Type} {store : TensorStore S Descriptor Value}
    {original output : Option Tensor} {initial current : S}
    (h : OptionMaterialized store original initial output current) : original.isSome=output.isSome := by
  cases original <;> cases output
  · rfl
  · exact False.elim h
  · exact False.elim h
  · rfl

theorem tensor_of_isSome (input : Option Tensor) (h : input.isSome=Bool.true) :
    ∃ tensor, input=some tensor := by
  cases input with
  | none => simp at h
  | some tensor => exact ⟨tensor,rfl⟩

theorem prepared_no_second_rowscale {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated) :
    ¬((p.flags q).x1=Bool.true ∧ (p.flags q).rowscale=Bool.true) := by
  rintro ⟨hx,hs⟩
  have hx' : q.x1.isSome=Bool.true := (optional_materialized_isSome h.inputs.x1).trans hx
  obtain ⟨input,hi⟩ := tensor_of_isSome q.x1 hx'
  have hnone := (hv.x1 input hi).2.1
  have hs' : q.rowscale.isSome=Bool.true := (optional_materialized_isSome h.inputs.rowscale).trans hs
  simp [hnone] at hs'

#axiomsClean materialized_of_some
#axiomsClean optional_materialized_isSome
#axiomsClean tensor_of_isSome
#axiomsClean prepared_no_second_rowscale
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
