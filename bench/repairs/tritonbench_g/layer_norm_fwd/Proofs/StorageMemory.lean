import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageAddress
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineMemory

namespace VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

/-- Guarded machine store using the actual descriptor width. The value is
already the result of the emitted destination conversion instruction. -/
def store {α : Type} (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) (value : Cell α) :
    TypedPhysical.Physical α (layout a) :=
  if machineGuard f N col role then fun i =>
    if i.val=nativeAddress a r stride M N row col role then value else physical i
  else physical

def load {α : Type} (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (f : Flags) (role : Role) (physical : TypedPhysical.Physical α (layout a)) : Option (Cell α) :=
  if machineGuard f N col role then
    if hb : nativeAddress a r stride M N row col role<2^64 then
      some (physical ⟨nativeAddress a r stride M N row col role,hb⟩)
    else none
  else none

theorem store_exact {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (physical : TypedPhysical.Physical α (layout a)) (value : Cell α) :
    store a r stride M N row col f role physical value=
      if active f role ∧ NativeAccess.laneActive N col role then
        TypedPhysical.write (layout a) physical (region r role,offset stride M N row col role) value
      else physical := by
  by_cases hg : machineGuard f N col role=Bool.true
  · have hl := (machine_guard_iff ha hc f role).mp hg
    rw [store,if_pos hg,if_pos hl]
    funext i
    rw [native_address_exact f r stride M N block a ha hd row col hr hc role hg]
    rfl
  · have hl : ¬(active f role ∧ NativeAccess.laneActive N col role) :=
      fun h => hg ((machine_guard_iff ha hc f role).mpr h)
    simp only [store,if_neg hg,if_neg hl]

theorem store_pull {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role) (hw : NativeMachineMemory.writeRole role)
    (physical : TypedPhysical.Physical α (layout a)) (value : Cell α) (template : State α) :
    TypedPhysical.pull (layout a) (store a r stride M N row col f role physical value) template=
      if active f role ∧ NativeAccess.laneActive N col role then
        (TypedPhysical.pull (layout a) physical template).write (region r role) (offset stride M N row col role) value
      else TypedPhysical.pull (layout a) physical template := by
  rw [store_exact f r stride M N block a ha hd row col hr hc role physical value]
  split_ifs with h
  · exact TypedPhysical.store_commutes (layout a) physical template (AliasMemory.Writes f r stride M N)
      hd.fresh _ (NativeMachineMemory.role_written f r stride M N row col hr role hw h.1 h.2) value
  · rfl

/-- Actual word-addressed load equals the current physical view at every
active source coordinate. It does not read a precomputed output or shadow copy. -/
theorem load_exact {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (ha : Accepted M N block) (hd : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (physical : TypedPhysical.Physical α (layout a)) (template : State α) :
    load a r stride M N row col f role physical=
      if active f role ∧ NativeAccess.laneActive N col role then
        some ((TypedPhysical.pull (layout a) physical template).mem (region r role) (offset stride M N row col role))
      else none := by
  by_cases hg : machineGuard f N col role=Bool.true
  · have hl := (machine_guard_iff ha hc f role).mp hg
    have hlive : (layout a).live (region r role,offset stride M N row col role) :=
      NativeAccess.all_roles_bounded f r stride M N row col _ (hd.accesses row hr) role hl.1 hl.2
    have he := native_address_exact f r stride M N block a ha hd row col hr hc role hg
    have hb : nativeAddress a r stride M N row col role<2^64 := by
      rw [he]
      exact (layout a).bounded _ hlive
    simp only [load,if_pos hg,dif_pos hb,if_pos hl,TypedPhysical.pull,TypedPhysical.read,dif_pos hlive]
    congr 2
    exact Fin.ext he
  · have hl : ¬(active f role ∧ NativeAccess.laneActive N col role) :=
      fun h => hg ((machine_guard_iff ha hc f role).mpr h)
    simp only [load,if_neg hg,if_neg hl]

#axiomsClean store_exact
#axiomsClean store_pull
#axiomsClean load_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
