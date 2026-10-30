import bench.tritonbench_g.block_sparse_attn.Output

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

theorem outputOffset_injective {α : Type} (p : Parameters) (s : State α)
    (stride : 16≤p.om) : Function.Injective (outputOffset p s) := by
  intro i j equal
  have smallerI : i.2.1.val<p.om := lt_of_lt_of_le i.2.1.isLt stride
  have smallerJ : j.2.1.val<p.om := lt_of_lt_of_le j.2.1.isLt stride
  have positive : 0<p.om := by omega
  have paired : row s i.1*p.om+i.2.1.val=row s j.1*p.om+j.2.1.val := by
    unfold outputOffset at equal
    omega
  have rows := congrArg (fun n => n/p.om) paired
  dsimp only at rows
  have divide (x y : Nat) (small : y<p.om) : (x*p.om+y)/p.om=x := by
    rw [Nat.mul_comm,Nat.mul_add_div positive,Nat.div_eq_of_lt small,Nat.add_zero]
  rw [divide _ _ smallerI,divide _ _ smallerJ] at rows
  have columns : i.2.1.val=j.2.1.val := by rw [rows] at paired; omega
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  simp only [row] at rows
  have hij : i=j := Fin.ext (by omega)
  have hkl : k=l := Fin.ext columns
  subst j; subst l; rfl

theorem outputMemory_readback {α : Type} (p : Parameters) (s : State (Replicas α))
    (values : Matrix α) (stride : 16≤p.om) (i : TileIndex [16,16])
    (active : row s i.1<p.length) :
    outputMemory p s values "out" (outputOffset p s i)=Cell.mk .real (values i) := by
  unfold outputMemory
  rw [← scatter_memory p s s values]
  exact State.masked_scatter_readback "out" (outputOffset p s) (fun i => .mk .real (values i))
    (fun i => row s i.1<p.length) s i active (outputOffset_injective p s stride)

theorem outputMemory_frame {α : Type} (p : Parameters) (s : State (Replicas α))
    (values : Matrix α) (region : RegionName) (address : Nat)
    (outside : region≠"out" ∨ ∀ i : TileIndex [16,16], row s i.1<p.length →
      address≠outputOffset p s i) : outputMemory p s values region address=s.mem region address := by
  unfold outputMemory
  rw [← scatter_memory p s s values]
  apply State.masked_scatter_frame
  rcases outside with outside | outside
  · exact Or.inl outside
  · exact Or.inr (fun i _ active => outside i active)

#axiomsClean outputOffset_injective
#axiomsClean outputMemory_readback
#axiomsClean outputMemory_frame
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
