import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperActiveCoordinates

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

/-- Tags identify source pointer roles. Aliasing is represented by their
physical descriptors, so distinct read tags may still share all their bytes. -/
def preparedRegions : Regions :=
  ⟨"X","Y","W","B","RESIDUAL","X1","W1","B1","Y1","RESIDUAL_OUT",
    "ROWSCALE","SEEDS","MASK","MEAN","RSTD"⟩

def regionTensor (p : Prepared) (region : RegionName) : Option Tensor :=
  match region.name with
  | "X" => some p.inputs.x | "W" => some p.inputs.weight
  | "RESIDUAL" => p.inputs.residual | "X1" => p.inputs.x1
  | "B" => p.inputs.bias | "W1" => p.inputs.weight1 | "B1" => p.inputs.bias1
  | "ROWSCALE" => p.inputs.rowscale
  | "Y" => some p.outputs.y | "Y1" => p.outputs.y1 | "RESIDUAL_OUT" => p.outputs.residualOut
  | "MEAN" => p.outputs.mean | "RSTD" => some p.outputs.rstd
  | "SEEDS" => p.outputs.seeds | "MASK" => p.outputs.mask
  | _ => none

def emptyDescriptor : Descriptor := ⟨0,0,4,by decide,by decide,by decide⟩

def preparedAllocations {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (p : Prepared) (region : RegionName) : Descriptor :=
  match regionTensor p region with
  | none => emptyDescriptor
  | some tensor => store.descriptor s tensor

theorem regionTensor_role (p : Prepared) (role : LayerNormFwd.Address.Role) :
    regionTensor p (LayerNormFwd.Address.region preparedRegions role)=roleTensor p role := by
  cases role <;> rfl

theorem preparedAllocation_role {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (p : Prepared) (role : LayerNormFwd.Address.Role)
    (tensor : Tensor) (he : roleTensor p role=some tensor) :
    preparedAllocations store s p (LayerNormFwd.Address.region preparedRegions role)=store.descriptor s tensor := by
  simp only [preparedAllocations,regionTensor_role,he]

theorem allocation_live_tensor {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (p : Prepared) (region : RegionName) (offset : Nat)
    (h : offset<(preparedAllocations store s p region).extent) :
    ∃ tensor, regionTensor p region=some tensor ∧
      preparedAllocations store s p region=store.descriptor s tensor := by
  cases he : regionTensor p region with
  | none => simp [preparedAllocations,he,emptyDescriptor] at h
  | some tensor => exact ⟨tensor,rfl,by simp [preparedAllocations,he]⟩

theorem prepared_output_names :
    [preparedRegions.y1,preparedRegions.y,preparedRegions.rstd,preparedRegions.mean,
      preparedRegions.residualOut,preparedRegions.dropoutMask].Nodup := by decide

theorem prepared_read_write_separated (f : Flags) : Domain.ReadWriteSeparated f preparedRegions := by
  intro input hi output ho he
  subst output
  rcases hi with hi | hi | ⟨_,hi⟩ | ⟨_,hi⟩ | ⟨_,hi⟩ | ⟨_,hi⟩ | ⟨_,hi⟩ | ⟨_,hi⟩ | ⟨_,_,hi⟩ <;>
    rcases ho with ho | ho | ⟨_,ho⟩ | ⟨_,ho⟩ | ⟨_,ho⟩ | ⟨_,_,ho⟩
  all_goals simp_all [preparedRegions]

/-- Allocation bounds come from the actual tensor coordinate, including
scalar roles at N=0. No conventional nonempty-span formula is needed. -/
theorem prepared_role_bounds {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role)
    (hl : NativeAccess.laneActive p.validated.N col role) :
    LayerNormFwd.Address.offset (argumentStrides args) p.validated.M p.validated.N row col role<
      (preparedAllocations store current p (LayerNormFwd.Address.region preparedRegions role)).extent := by
  obtain ⟨value,he,hp,_,hc,hindex⟩ :=
    active_role_coordinate store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  rw [preparedAllocation_role store current p role value he,hindex]
  exact hm.bounds current value hp _ hc

theorem prepared_row_bounds {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs) (row : Nat) (hr : row<p.validated.M) :
    FlatDomain.RowBounds (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N row
      (fun region => (preparedAllocations store current p region).extent) := by
  have bound := fun col role ha hl =>
    prepared_role_bounds store hm tensor q initial current p args h hv hi ho row col hr role ha hl
  exact {
    x := fun col hc => bound col .x trivial hc
    y := fun col hc => bound col .y trivial hc
    weight := fun col hc => bound col .weight trivial hc
    rstd := bound 0 .rstd trivial trivial
    bias := fun hf col hc => bound col .bias hf hc
    mean := fun hf => bound 0 .mean hf trivial
    residual := fun hf col hc => bound col .residual hf hc
    residualOut := fun hf col hc => bound col .residualOut hf hc
    x1 := fun hf col hc => bound col .x1 hf hc
    weight1 := fun hf col hc => bound col .weight1 hf hc
    bias1 := fun hw hb col hc => bound col .bias1 ⟨hw,hb⟩ hc
    y1 := fun hf col hc => bound col .y1 hf hc
    rowscale := fun hf => bound 0 .rowscale hf trivial
    rowscale1 := fun hx hs => bound 0 .rowscale1 ⟨hx,hs⟩ trivial
    seeds := fun hf => bound 0 .seeds hf trivial
    seeds1 := fun hx hs => bound 0 .seeds1 ⟨hx,hs⟩ trivial
    mask := fun hd hm col hc => bound col .mask ⟨hd,hm⟩ hc
    mask1 := fun hx hd hm col hc => bound col .mask1 ⟨hx,hd,hm⟩ hc }

#axiomsClean regionTensor_role
#axiomsClean preparedAllocation_role
#axiomsClean allocation_live_tensor
#axiomsClean prepared_output_names
#axiomsClean prepared_read_write_separated
#axiomsClean prepared_role_bounds
#axiomsClean prepared_row_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
