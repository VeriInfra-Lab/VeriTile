import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLaunchState
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperValidation

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000

/-- Ordinary selected runtime scalar conversion, taking the full host binary64
word. This interface is total, including NaNs, infinities and signed zero;
it contains no FP-to-Real exactness law. -/
structure ScalarEncoding (α : Type) where
  fromHostBits : UInt64 → α

def ScalarEncoding.encode (encoding : ScalarEncoding α) (value : Float) : α :=
  encoding.fromHostBits value.toBits

def argumentStrides (args : LaunchArguments) : Strides where
  x := args.strideX.toNat
  y := args.strideY.toNat
  residual := args.strideResidual.toNat
  residualOut := args.strideResidualOut.toNat
  x1 := args.strideX1.toNat
  y1 := args.strideY1.toNat

/-- Config fields are constructed from actual launch arguments. Allocation
coverage and device admission are separate ordinary program obligations. -/
def configOfArguments (ops : NativeRow.Ops α) (encoding : ScalarEncoding α)
    (args : LaunchArguments) (red : Nat → Whole.Reductions args.prepared.block)
    (plan plan1 : NativeRow.AffinePlan) (regions : Regions) (allocations : RegionName → Descriptor)
    (ha : NativeABI.Accepted args.prepared.validated.M args.prepared.validated.N args.prepared.block)
    (hs : PhysicalDomain.Domain (args.prepared.flags args.request) regions (argumentStrides args)
      args.prepared.validated.M args.prepared.validated.N allocations) : Config α args.prepared.block where
  ops := ops
  reductions := red
  plan := plan
  plan1 := plan1
  flags := args.prepared.flags args.request
  regions := regions
  strides := argumentStrides args
  M := args.prepared.validated.M
  N := args.prepared.validated.N
  eps := encoding.encode args.request.eps
  p := encoding.encode args.request.dropoutP
  allocations := allocations
  accepted := ha
  storage := hs

structure ArgumentsMatch (encoding : ScalarEncoding α) (args : LaunchArguments)
    (c : Config α args.prepared.block) : Prop where
  flags : c.flags=args.prepared.flags args.request
  strides : c.strides=argumentStrides args
  rows : c.M=args.prepared.validated.M
  cols : c.N=args.prepared.validated.N
  eps : c.eps=encoding.fromHostBits args.request.eps.toBits
  dropoutP : c.p=encoding.fromHostBits args.request.dropoutP.toBits

theorem configOfArguments_match (ops : NativeRow.Ops α) (encoding : ScalarEncoding α)
    (args : LaunchArguments) (red : Nat → Whole.Reductions args.prepared.block)
    (plan plan1 : NativeRow.AffinePlan) (regions : Regions) (allocations : RegionName → Descriptor)
    (ha : NativeABI.Accepted args.prepared.validated.M args.prepared.validated.N args.prepared.block)
    (hs : PhysicalDomain.Domain (args.prepared.flags args.request) regions (argumentStrides args)
      args.prepared.validated.M args.prepared.validated.N allocations) :
    ArgumentsMatch encoding args (configOfArguments ops encoding args red plan plan1 regions allocations ha hs) :=
  ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem optional_materialized_present {S Value : Type}
    {store : TensorStore S Descriptor Value} {original output : Option Tensor} {initial current : S}
    (h : OptionMaterialized store original initial output current) : OptionPresent store current output := by
  cases original <;> cases output
  · trivial
  · exact False.elim h
  · trivial
  · exact h.present

theorem optional_allocated_present {S Value : Type}
    {store : TensorStore S Descriptor Value} {condition : Bool} {request : Wrapper.Allocation}
    {output : Option Tensor} {current : S}
    (h : OptionAllocated store condition request output current) : OptionPresent store current output := by
  cases condition <;> cases output
  · trivial
  · exact False.elim h
  · trivial
  · exact h.present

theorem tensor_stride_range {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor)
    (value : Int) (hv : tensor.strides[0]?=some value) :
    0≤value ∧ value<2^63 ∧ value.toNat<2^63 := by
  have hr := (hm.strides s tensor hp).2 value (List.mem_of_getElem? hv)
  exact ⟨hr.1,hr.2,by omega⟩

theorem optional_stride_range {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (s : HostState α Host) (input : Option Tensor) (hp : OptionPresent store s input)
    (value : Int) (hv : OptionalStrideBound input value) :
    0≤value ∧ value<2^63 ∧ value.toNat<2^63 := by
  cases input with
  | none => have he : value=0 := hv; simp [he]
  | some tensor => exact tensor_stride_range store hm s tensor hp value hv

/-- Signed tensor metadata is derived for all six actual launch strides,
including source zeros for absent optional tensors. -/
theorem launch_argument_metadata {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated) :
    PhysicalDomain.Metadata (argumentStrides args) p.validated.M p.validated.N := by
  have hx : p.inputs.x.shape=[p.validated.M,p.validated.N] := h.inputs.x.shape.trans hv.x.1
  have hmrows := hm.shape current p.inputs.x h.inputs.x.present p.validated.M (by simp [hx])
  have hmcols := hm.shape current p.inputs.x h.inputs.x.present p.validated.N (by simp [hx])
  exact ⟨hmrows,hmcols,
    (tensor_stride_range store hm current p.inputs.x h.inputs.x.present _ h.arguments.x).2.2,
    (tensor_stride_range store hm current p.outputs.y h.outputs.1.present _ h.arguments.y).2.2,
    (optional_stride_range store hm current p.inputs.residual
      (optional_materialized_present h.inputs.residual) _ h.arguments.residual).2.2,
    (optional_stride_range store hm current p.outputs.residualOut
      (optional_allocated_present h.outputs.2.2.1) _ h.arguments.residualOut).2.2,
    (optional_stride_range store hm current p.inputs.x1
      (optional_materialized_present h.inputs.x1) _ h.arguments.x1).2.2,
    (optional_stride_range store hm current p.outputs.y1
      (optional_allocated_present h.outputs.2.1) _ h.arguments.y1).2.2⟩

#axiomsClean configOfArguments_match
#axiomsClean optional_materialized_present
#axiomsClean optional_allocated_present
#axiomsClean tensor_stride_range
#axiomsClean optional_stride_range
#axiomsClean launch_argument_metadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
