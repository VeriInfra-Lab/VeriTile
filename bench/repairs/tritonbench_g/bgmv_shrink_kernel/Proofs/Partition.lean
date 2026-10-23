import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.RealExecution

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Partition
open VeriTile.Triton Geometry
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Sum exactly the independent K coordinates. This theorem is Real-only;
it never reassociates any floating-point operation. -/
theorem sum_partition (h : Call) (f : Nat → ℝ) :
    (∑ s : Fin 64,∑ c ∈ Finset.range h.cycles,∑ e : Fin 256,
      if rank s.val c e.val<h.K then f (rank s.val c e.val) else 0)=∑ k : Fin h.K,f k.val := by
  classical
  let T := Fin 64 × Fin h.cycles × Fin 256
  let key : T → Nat := fun t => rank t.1.val t.2.1.val t.2.2.val
  have bij : (∑ t ∈ Finset.univ.filter (fun t : T => key t<h.K),f (key t))=
      ∑ k : Fin h.K,f k.val := by
    apply Finset.sum_bij (fun t ht => (⟨key t,(Finset.mem_filter.mp ht).2⟩ : Fin h.K))
    · intro t ht; exact Finset.mem_univ _
    · intro t ht u hu same
      have kt : key t<h.K := (Finset.mem_filter.mp ht).2
      obtain ⟨v,hv,unique⟩ := rank_coverage h ⟨key t,kt⟩
      exact (unique t rfl).trans (unique u (congrArg Fin.val same).symm).symm
    · intro k _
      obtain ⟨t,ht,_⟩ := rank_coverage h k
      have inside : key t<h.K := by dsimp [key]; rw [ht]; exact k.isLt
      exact ⟨t,Finset.mem_filter.mpr ⟨Finset.mem_univ _,inside⟩,Fin.ext ht⟩
    · intro t ht; rfl
  rw [Finset.sum_filter] at bij
  calc
    _ = ∑ s : Fin 64,∑ c : Fin h.cycles,∑ e : Fin 256,
        if rank s.val c.val e.val<h.K then f (rank s.val c.val e.val) else 0 := by
      apply Finset.sum_congr rfl
      intro s _
      exact (Fin.sum_univ_eq_sum_range (fun c => ∑ e : Fin 256,
        if rank s.val c e.val<h.K then f (rank s.val c e.val) else 0) h.cycles).symm
    _ = _ := by simpa only [T,key,Fintype.sum_prod_type] using bij

theorem contributions (h : Call) (u : BlockState) (b slot n : Nat) :
    (∑ s : Fin 64,RealExecution.contribution h u b slot n s.val)=
      h.scaling*∑ k : Fin h.K,u.readMem "X" (inputOffset h b k.val)*
        u.readMem "W" (weightOffset h slot n k.val) := by
  simp only [RealExecution.contribution,←Finset.mul_sum]
  exact congrArg (h.scaling * ·) (sum_partition h (fun k =>
    u.readMem "X" (inputOffset h b k)*u.readMem "W" (weightOffset h slot n k)))

#axiomsClean sum_partition
#axiomsClean contributions
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Partition
