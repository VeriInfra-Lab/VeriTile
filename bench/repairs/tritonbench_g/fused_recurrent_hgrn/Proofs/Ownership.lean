import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Whole

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section

theorem block_decode (block BD : Nat) (hBD : 0 < BD) (j : Fin BD) :
    (block*BD+j.val)/BD=block := by
  simp [Nat.add_div,Nat.div_eq_of_lt j.isLt,Nat.mod_eq_of_lt j.isLt,Nat.not_le.mpr j.isLt,hBD]

def outputOwner (T D BD : Nat) (r : RegionName) (a : Nat) : Program :=
  ((a%D)/BD,if r="ht" then a/D else a/D/T)

theorem forward_owner (T D BD : Nat) (hBD : 0 < BD) (S : Bool) (id : Program) (r : RegionName) (a : Nat)
    (h : fWrites T D BD S id r a) : id=outputOwner T D BD r a := by
  rcases h with ⟨rfl,t,ht,j,hj,rfl⟩|⟨_,rfl,j,hj,rfl⟩
  · have dec := row_decode id.2 T D t (id.1*BD+j.val) ht hj
    simp [outputOwner,dec.1,dec.2.2,block_decode id.1 BD hBD j]
  · have dec := slot_decode id.2 D (id.1*BD+j.val) hj
    simp [outputOwner,dec.1,dec.2,block_decode id.1 BD hBD j]

theorem backward_owner (T D BD : Nat) (hBD : 0 < BD) (id : Program) (r : RegionName) (a : Nat)
    (h : bWrites T D BD id r a) : id=outputOwner T D BD r a := by
  obtain ⟨hr,t,ht,j,hj,rfl⟩ := h
  have dec := row_decode id.2 T D t (id.1*BD+j.val) ht hj
  rcases hr with rfl|rfl <;>
    simp [outputOwner,dec.1,dec.2.2,block_decode id.1 BD hBD j]

/-- Distinct programs cannot write the same cell. Inputs are framed separately
by the complete per-program theorems, so this is also the race-free grid boundary. -/
theorem forward_unique (T D BD : Nat) (hBD : 0 < BD) (S : Bool)
    (id other : Program) (r : RegionName) (a : Nat)
    (h : fWrites T D BD S id r a) (ho : fWrites T D BD S other r a) : id=other :=
  (forward_owner T D BD hBD S id r a h).trans (forward_owner T D BD hBD S other r a ho).symm

theorem backward_unique (T D BD : Nat) (hBD : 0 < BD)
    (id other : Program) (r : RegionName) (a : Nat)
    (h : bWrites T D BD id r a) (ho : bWrites T D BD other r a) : id=other :=
  (backward_owner T D BD hBD id r a h).trans (backward_owner T D BD hBD other r a ho).symm

#axiomsClean forward_unique
#axiomsClean backward_unique
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
