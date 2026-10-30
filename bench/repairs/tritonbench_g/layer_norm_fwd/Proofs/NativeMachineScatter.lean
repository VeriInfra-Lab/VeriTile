import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineMemory
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeSchedule

/-! Instruction-level finite-width vector/scalar store simulation. Source
masked tile stores execute their actual lane sequence; inactive lanes retain
the whole physical state. No final-output reconstruction is used here. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachineMemory
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def scatter {α : Type} (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row block : Nat) (f : Flags) (role : Role) (values : Fin block → Cell α)
    (physical : TypedPhysical.Physical α (layout a)) : TypedPhysical.Physical α (layout a) :=
  (TileShape.allIndices [block]).foldl (fun p j =>
    store a r stride M N row j.1.val f role p (values j.1)) physical

theorem sequence_commutes {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row : Nat) (hr : row<M) (role : Role) (hw : writeRole role)
    (values : Fin block → Cell α) (indices : List (TileIndex [block]))
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    TypedPhysical.pull (layout a)
      (indices.foldl (fun p j => store a r stride M N row j.1.val f role p (values j.1)) physical) template=
      indices.foldl (fun s j => if active f role ∧ NativeAccess.laneActive N j.1.val role then
        s.write (region r role) (offset stride M N row j.1.val role) (values j.1) else s)
        (TypedPhysical.pull (layout a) physical template) := by
  induction indices generalizing physical template with
  | nil => rfl
  | cons j js ih =>
    simp only [List.foldl_cons]
    rw [ih,store_commutes f r stride M N block a ha hd row j.1.val hr j.1.isLt role hw physical (values j.1) template]

theorem vector_lane_active (role : Role) (h : scalarRole role=Bool.false) (N col : Nat) :
    NativeAccess.laneActive N col role ↔ col<N := by
  cases role <;> simp_all [scalarRole,NativeAccess.laneActive]

theorem scalar_lane_active (role : Role) (h : scalarRole role=Bool.true) (N col : Nat) :
    NativeAccess.laneActive N col role := by
  cases role <;> simp_all [scalarRole,NativeAccess.laneActive]

theorem scatter_commutes {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row : Nat) (hr : row<M) (role : Role) (hw : writeRole role) (hactive : active f role)
    (hvector : scalarRole role=Bool.false) (base : Nat)
    (hoffset : ∀ col, offset stride M N row col role=base+col)
    (values : Fin block → Cell α) (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    TypedPhysical.pull (layout a) (scatter a r stride M N row block f role values physical) template=
      NativeSchedule.vectorStore (TypedPhysical.pull (layout a) physical template) (region r role) base N block values := by
  rw [scatter,sequence_commutes f r stride M N block a ha hd row hr role hw]
  unfold NativeSchedule.vectorStore
  congr 1
  funext s j
  simp only [hactive,true_and,vector_lane_active role hvector,hoffset]

theorem scalar_store_commutes {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row : Nat) (hr : row<M) (role : Role) (hw : writeRole role) (hactive : active f role)
    (hscalar : scalarRole role=Bool.true) (physical : TypedPhysical.Physical α (layout a))
    (value : Cell α) (template : State α) :
    TypedPhysical.pull (layout a) (store a r stride M N row 0 f role physical value) template=
      NativeSchedule.scalarStore (TypedPhysical.pull (layout a) physical template)
        (region r role) (offset stride M N row 0 role) value := by
  rw [store_commutes f r stride M N block a ha hd row 0 hr ha.blockPositive role hw physical value template]
  rw [if_pos ⟨hactive,scalar_lane_active role hscalar N 0⟩]
  rfl

/-- N=0 does not suppress the row's rstd store. This is an actual store-state
simulation, stronger than the corresponding standalone address equation. -/
theorem zero_columns_rstd_store {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M 0 block) (hd : PhysicalDomain.Domain f r stride M 0 a)
    (row : Nat) (hr : row<M) (physical : TypedPhysical.Physical α (layout a))
    (value : Cell α) (template : State α) :
    TypedPhysical.pull (layout a) (store a r stride M 0 row 0 f .rstd physical value) template=
      (TypedPhysical.pull (layout a) physical template).write r.rstd row value :=
  scalar_store_commutes f r stride M 0 block a ha hd row hr .rstd trivial trivial rfl physical value template

#axiomsClean sequence_commutes
#axiomsClean scatter_commutes
#axiomsClean scalar_store_commutes
#axiomsClean zero_columns_rstd_store
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachineMemory
