import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineScatter

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachineMemory
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI NativeRow
noncomputable section
set_option maxHeartbeats 2000000

def loaded {α : Type} (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) : Cell α :=
  (load a r stride M N row col f role physical).getD (.mk .nat 0)

def raw {α : Type} [Inhabited α] (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) : Expr α :=
  .cast32 (if col<N then .input ((loaded a r stride M N row col f role physical).read .real) else .literal 0)

def scalar {α : Type} [Inhabited α] (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) : Expr α :=
  .cast32 (.input ((loaded a r stride M N row 0 f role physical).read .real))

theorem loaded_eq_tagged {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (hactive : active f role) (hlane : NativeAccess.laneActive N col role)
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    loaded a r stride M N row col f role physical=
      (TypedPhysical.pull (layout a) physical template).mem (region r role) (offset stride M N row col role) := by
  rw [loaded,load_eq_tagged f r stride M N block a ha hd row col hr hc role physical template,if_pos ⟨hactive,hlane⟩]
  rfl

/-- Source-style raw load/cast graph from a word-addressed physical read.
Padding is the literal zero graph, and no inactive lane reads a physical cell. -/
theorem raw_correspondence {α : Type} [Inhabited α] (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role) (hactive : active f role)
    (hvector : scalarRole role=Bool.false) (base : Nat)
    (hoffset : offset stride M N row col role=base+col)
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    raw a r stride M N row col f role physical=
      NativeSchedule.raw (TypedPhysical.pull (layout a) physical template) (region r role) base N col := by
  by_cases hcol : col<N
  · rw [raw,if_pos hcol,loaded_eq_tagged f r stride M N block a ha hd row col hr hc role hactive
      ((vector_lane_active role hvector N col).mpr hcol) physical template,hoffset]
    simp only [NativeSchedule.raw,if_pos hcol]
  · simp only [raw,NativeSchedule.raw,if_neg hcol]

/-- Unmasked scalar loads use lane zero even for N=0. Admission establishes
a nonempty compile-time tile, independently of the active vector width. -/
theorem scalar_correspondence {α : Type} [Inhabited α] (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row : Nat) (hr : row<M) (role : Role) (hactive : active f role) (hscalar : scalarRole role=Bool.true)
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    scalar a r stride M N row f role physical=
      NativeSchedule.scalar (TypedPhysical.pull (layout a) physical template) (region r role) (offset stride M N row 0 role) := by
  rw [scalar,loaded_eq_tagged f r stride M N block a ha hd row 0 hr ha.blockPositive role hactive
    (scalar_lane_active role hscalar N 0) physical template]
  rfl

theorem zero_columns_seed_load {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M 0 block) (hd : PhysicalDomain.Domain f r stride M 0 a)
    (row : Nat) (hr : row<M) (hdrop : f.dropout=Bool.true)
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    load a r stride M 0 row 0 f .seeds physical=
      some ((TypedPhysical.pull (layout a) physical template).mem r.seeds row) := by
  rw [load_eq_tagged f r stride M 0 block a ha hd row 0 hr ha.blockPositive .seeds physical template]
  simp [active,NativeAccess.laneActive,hdrop,region,offset]

#axiomsClean loaded_eq_tagged
#axiomsClean raw_correspondence
#axiomsClean scalar_correspondence
#axiomsClean zero_columns_seed_load
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachineMemory
