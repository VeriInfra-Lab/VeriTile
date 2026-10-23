import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeABI
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedPhysical

/-! Guarded fixed-width physical loads/stores, composed with the unchanged
tagged-cell memory model. These lemmas concern actual cell reads/writes, not
merely equal addresses. The carrier is typed cells at byte starts; complete
access spans are separately bounded by NativeABI.native_access_span. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachineMemory
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def load {α : Type} (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) : Option (Cell α) :=
  if machineGuard f N col role then
    if hb : nativeAddress a r stride M N row col role<2^64 then
      some (physical ⟨nativeAddress a r stride M N row col role,hb⟩)
    else none
  else none

def store {α : Type} (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) (value : Cell α) :
    TypedPhysical.Physical α (layout a) :=
  if machineGuard f N col role then fun i =>
    if i.val=nativeAddress a r stride M N row col role then value else physical i
  else physical

def writeRole : Role → Prop
  | .y | .y1 | .residualOut | .mean | .rstd | .mask | .mask1 => True
  | _ => False

theorem role_live (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (hg : machineGuard f N col role=Bool.true) :
    (layout a).live (region r role,offset stride M N row col role) := by
  obtain ⟨hactive,hlane⟩ := (machine_guard_iff ha hc f role).mp hg
  exact NativeAccess.all_roles_bounded f r stride M N row col _
    (hd.accesses row hr) role hactive hlane

theorem role_written (f : Flags) (r : Regions) (stride : Strides) (M N row col : Nat)
    (hr : row<M) (role : Role) (hw : writeRole role) (ha : active f role)
    (hl : NativeAccess.laneActive N col role) :
    AliasMemory.Writes f r stride M N (region r role,offset stride M N row col role) := by
  refine ⟨row,hr,?_⟩
  cases role <;> simp_all [writeRole,active,NativeAccess.laneActive,region,offset,Contract.written,Contract.inRow_of_col]

theorem load_eq_tagged {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    load a r stride M N row col f role physical=
      if active f role ∧ NativeAccess.laneActive N col role then
        some ((TypedPhysical.pull (layout a) physical template).mem (region r role) (offset stride M N row col role))
      else none := by
  by_cases hg : machineGuard f N col role=Bool.true
  · have hlogical := (machine_guard_iff ha hc f role).mp hg
    have hlive := role_live f r stride M N block a ha hd row col hr hc role hg
    have he := native_address_exact f r stride M N block a ha hd row col hr hc role hg
    have hb : nativeAddress a r stride M N row col role<2^64 := by
      rw [he]
      exact (layout a).bounded _ hlive
    simp only [load,if_pos hg,dif_pos hb,if_pos hlogical,TypedPhysical.pull,TypedPhysical.read,dif_pos hlive]
    congr 2
    exact Fin.ext he
  · have hlogical : ¬(active f role ∧ NativeAccess.laneActive N col role) :=
      fun h => hg ((machine_guard_iff ha hc f role).mpr h)
    simp only [load,if_neg hg,if_neg hlogical]

theorem store_eq_physical {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (physical : TypedPhysical.Physical α (layout a)) (value : Cell α) :
    store a r stride M N row col f role physical value=
      if active f role ∧ NativeAccess.laneActive N col role then
        TypedPhysical.write (layout a) physical (region r role,offset stride M N row col role) value
      else physical := by
  by_cases hg : machineGuard f N col role=Bool.true
  · have hlogical := (machine_guard_iff ha hc f role).mp hg
    rw [store,if_pos hg,if_pos hlogical]
    funext i
    rw [native_address_exact f r stride M N block a ha hd row col hr hc role hg]
    rfl
  · have hlogical : ¬(active f role ∧ NativeAccess.laneActive N col role) :=
      fun h => hg ((machine_guard_iff ha hc f role).mpr h)
    simp only [store,if_neg hg,if_neg hlogical]

/-- One guarded source-role store with parameter decoding, pid/lane widths,
mask/control and physical byte address already composed. It preserves all
other tagged cells, including aliases, by the ordinary fresh-output domain. -/
theorem store_commutes {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role) (hw : writeRole role)
    (physical : TypedPhysical.Physical α (layout a)) (value : Cell α) (template : State α) :
    TypedPhysical.pull (layout a) (store a r stride M N row col f role physical value) template=
      if active f role ∧ NativeAccess.laneActive N col role then
        (TypedPhysical.pull (layout a) physical template).write (region r role) (offset stride M N row col role) value
      else TypedPhysical.pull (layout a) physical template := by
  rw [store_eq_physical f r stride M N block a ha hd row col hr hc role physical value]
  split_ifs with h
  · exact TypedPhysical.store_commutes (layout a) physical template (AliasMemory.Writes f r stride M N)
      hd.fresh _ (role_written f r stride M N row col hr role hw h.1 h.2) value
  · rfl

#axiomsClean role_live
#axiomsClean role_written
#axiomsClean load_eq_tagged
#axiomsClean store_eq_physical
#axiomsClean store_commutes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachineMemory
