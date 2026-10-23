import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelStore
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelStatic
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelStateMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelMemory
noncomputable section
variable {α : Type} [Inhabited α]

def store (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : ForwardTile s α) (u : State α) : State α :=
  (TileShape.allIndices [s.FBV,s.BK]).foldl (fun v ji =>
    if KernelStatic.mask (α:=α) s iv ji then
      v.write r (stateAddress s bh iv s.FBV ji.2.1.val ji.1.val) (KernelReference.stored M (value ji)) else v) u

theorem cell (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : ForwardTile s α) (u : State α) (i : Fin s.BK) (j : Fin s.FBV)
    (hi : i.val<s.K) (hj : iv*s.FBV+j.val<s.V) :
    (store M s r bh iv value u).mem r (stateAddress s bh iv s.FBV i.val j.val)=
      KernelReference.stored M (value (j,i,PUnit.unit)) := by
  apply masked_readback r (fun ji : TileIndex [s.FBV,s.BK] => stateAddress s bh iv s.FBV ji.2.1.val ji.1.val)
    (fun ji => KernelReference.stored M (value ji)) (KernelStatic.mask (α:=α) s iv)
    (TileShape.allIndices [s.FBV,s.BK]) u (j,i,PUnit.unit)
    (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ _) (by simp [KernelStatic.mask,hi,hj])
  rintro ⟨j',i',⟨⟩⟩ active same
  have both : i'.val<s.K ∧ iv*s.FBV+j'.val<s.V := by simpa [KernelStatic.mask] using active
  have active' := both.2
  have eq : i.val*s.V+(iv*s.FBV+j.val)=i'.val*s.V+(iv*s.FBV+j'.val) := by
    dsimp only [stateAddress] at same
    omega
  obtain ⟨hk,hv⟩ := Geometry.mixed_injective s.V i.val i'.val _ _ hj active' eq
  have key : i=i' := Fin.ext hk
  have val : j=j' := Fin.ext (by omega)
  simp [key,val]

theorem frame (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : ForwardTile s α) (u : State α) (r' : RegionName) (off : Nat)
    (outside : r'≠r ∨ ∀ (i : Fin s.BK) (j : Fin s.FBV),i.val<s.K → iv*s.FBV+j.val<s.V →
      off≠stateAddress s bh iv s.FBV i.val j.val) :
    (store M s r bh iv value u).mem r' off=u.mem r' off := by
  apply State.masked_scatter_frame
  rcases outside with different | miss
  · exact Or.inl different
  · right
    intro ji _ active
    have both : ji.2.1.val<s.K ∧ iv*s.FBV+ji.1.val<s.V := by simpa [KernelStatic.mask] using active
    exact miss ji.2.1 ji.1 both.1 both.2

theorem mem_congr (M : Algebra α) (s : Shape) (r : RegionName) (bh iv : Nat)
    (value : ForwardTile s α) (u v : State α) (same : u.mem=v.mem) :
    (store M s r bh iv value u).mem=(store M s r bh iv value v).mem :=
  KernelStore.masked_mem _ _ _ _ _ _ _ same

#axiomsClean cell
#axiomsClean frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelStateMemory
