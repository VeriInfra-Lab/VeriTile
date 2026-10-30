import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelReference
import bench.tritonbench_g.fused_recurrent_delta.Proofs.Geometry
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

/-- Only active offsets must be injective. Padded two-dimensional source
offsets may coincide and cannot justify the stronger unmasked premise. -/
theorem masked_readback {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Bool) (xs : List ι) (u : State α)
    (i : ι) (nd : xs.Nodup) (member : i∈xs) (enabled : active i=Bool.true)
    (unique : ∀ j,active j=Bool.true → off i=off j → i=j) :
    (xs.foldl (fun v j => if active j then v.write r (off j) (value j) else v) u).mem r (off i)=value i := by
  obtain ⟨before,after,eq⟩ := List.append_of_mem member
  rw [eq,List.nodup_append,List.nodup_cons] at nd
  obtain ⟨_,⟨absent,_⟩,_⟩ := nd
  rw [eq,List.foldl_append,List.foldl_cons,if_pos enabled]
  rw [State.masked_scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j member enabled same
    exact absent ((unique j enabled same) ▸ member)

def rowStore (M : Algebra α) (s : Shape) (r : RegionName) (bh iv BV t : Nat)
    (values : Fin BV → α) (u : State α) : State α :=
  (TileShape.allIndices [BV]).foldl (fun v j =>
    if decide (iv*BV+j.1.val<s.V) then
      v.write r (valAddress s bh iv BV t j.1.val) (KernelReference.stored M (values j.1)) else v) u

theorem rowStore_cell (M : Algebra α) (s : Shape) (r : RegionName) (bh iv BV t : Nat)
    (values : Fin BV → α) (u : State α) (j : Fin BV) (hj : iv*BV+j.val<s.V) :
    (rowStore M s r bh iv BV t values u).mem r (valAddress s bh iv BV t j.val)=
      KernelReference.stored M (values j) := by
  apply masked_readback r (fun j : TileIndex [BV] => valAddress s bh iv BV t j.1.val)
    (fun j => KernelReference.stored M (values j.1)) (fun j => decide (iv*BV+j.1.val<s.V))
    (TileShape.allIndices [BV]) u (j,PUnit.unit)
    (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ _) (by simpa)
  rintro ⟨k,⟨⟩⟩ _ same
  have equal : j=k := Fin.ext (by simpa [valAddress] using same)
  simp [equal]

theorem rowStore_frame (M : Algebra α) (s : Shape) (r : RegionName) (bh iv BV t : Nat)
    (values : Fin BV → α) (u : State α) (r' : RegionName) (off : Nat)
    (outside : r'≠r ∨ ∀ j : Fin BV,iv*BV+j.val<s.V → off≠valAddress s bh iv BV t j.val) :
    (rowStore M s r bh iv BV t values u).mem r' off=u.mem r' off := by
  apply State.masked_scatter_frame
  rcases outside with different | miss
  · exact Or.inl different
  · right
    intro j _ enabled
    exact miss j.1 (by simpa using enabled)

theorem rowStore_other_time (M : Algebra α) (s : Shape) (r : RegionName) (bh iv BV t n : Nat)
    (values : Fin BV → α) (u : State α) (j : Fin BV)
    (hj : iv*BV+j.val<s.V) (different : n≠t) :
    (rowStore M s r bh iv BV t values u).mem r (valAddress s bh iv BV n j.val)=
      u.mem r (valAddress s bh iv BV n j.val) := by
  apply rowStore_frame
  right
  intro k hk same
  have equal : n*s.V+(iv*BV+j.val)=t*s.V+(iv*BV+k.val) := by
    dsimp only [valAddress] at same
    omega
  exact different (Geometry.mixed_injective s.V n t _ _ hj hk equal).1

#axiomsClean masked_readback
#axiomsClean rowStore_cell
#axiomsClean rowStore_frame
#axiomsClean rowStore_other_time
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelMemory
