import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelStore
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardRegisters
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardStateMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelMemory
noncomputable section
variable {α : Type} [Inhabited α]

def mask (s : Shape) (iv : Nat) : Values α .bool [s.BK,s.BBV] :=
  fun ij => decide (ij.1.val<s.K) && decide (iv*s.BBV+ij.2.1.val<s.V)

def store (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : Tile s α) (u : State α) : State α :=
  (TileShape.allIndices [s.BK,s.BBV]).foldl (fun v ji =>
    if mask (α:=α) s iv ji then
      v.write r (stateAddress s bh iv s.BBV ji.1.val ji.2.1.val) (KernelReference.stored M (value ji)) else v) u

theorem cell (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : Tile s α) (u : State α) (i : Fin s.BK) (j : Fin s.BBV)
    (hi : i.val<s.K) (hj : iv*s.BBV+j.val<s.V) :
    (store M s r bh iv value u).mem r (stateAddress s bh iv s.BBV i.val j.val)=
      KernelReference.stored M (value (i,j,PUnit.unit)) := by
  apply masked_readback r (fun ji : TileIndex [s.BK,s.BBV] => stateAddress s bh iv s.BBV ji.1.val ji.2.1.val)
    (fun ji => KernelReference.stored M (value ji)) (mask (α:=α) s iv)
    (TileShape.allIndices [s.BK,s.BBV]) u (i,j,PUnit.unit)
    (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ _) (by simp [mask,hi,hj])
  rintro ⟨i',j',⟨⟩⟩ active same
  have both : i'.val<s.K ∧ iv*s.BBV+j'.val<s.V := by simpa [mask] using active
  have active' := both.2
  have eq : i.val*s.V+(iv*s.BBV+j.val)=i'.val*s.V+(iv*s.BBV+j'.val) := by
    dsimp only [stateAddress] at same
    omega
  obtain ⟨hk,hv⟩ := Geometry.mixed_injective s.V i.val i'.val _ _ hj active' eq
  have key : i=i' := Fin.ext hk
  have val : j=j' := Fin.ext (by omega)
  simp [key,val]

theorem frame (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : Tile s α) (u : State α) (r' : RegionName) (off : Nat)
    (outside : r'≠r ∨ ∀ (i : Fin s.BK) (j : Fin s.BBV),i.val<s.K → iv*s.BBV+j.val<s.V →
      off≠stateAddress s bh iv s.BBV i.val j.val) :
    (store M s r bh iv value u).mem r' off=u.mem r' off := by
  apply State.masked_scatter_frame
  rcases outside with different | miss
  · exact Or.inl different
  · right
    intro ji _ active
    have both : ji.1.val<s.K ∧ iv*s.BBV+ji.2.1.val<s.V := by simpa [mask] using active
    exact miss ji.1 ji.2.1 both.1 both.2

theorem mem_congr (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : Tile s α) (u v : State α) (same : u.mem=v.mem) :
    (store M s r bh iv value u).mem=(store M s r bh iv value v).mem :=
  KernelStore.masked_mem _ _ _ _ _ _ _ same

#axiomsClean cell
#axiomsClean frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardStateMemory
