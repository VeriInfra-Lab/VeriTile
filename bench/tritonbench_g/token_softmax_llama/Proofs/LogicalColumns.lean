import bench.tritonbench_g.token_softmax_llama.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.LogicalColumns
open VeriTile.Triton
noncomputable section

/-- Each covered logical segment contains exactly the active block lanes. -/
theorem sum_eq (n B : Nat) (cover : n ≤ B) (f : Fin B → ℝ) :
    (∑ j : Fin B, if j.val < n then f j else 0) =
      ∑ j : Fin n, if h : j.val < B then f ⟨j.val,h⟩ else 0 := by
  classical
  rw [← Finset.sum_filter]
  symm
  refine Finset.sum_bij (fun i _ => Fin.castLE cover i) ?_ ?_ ?_ ?_
  · intro i _
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, i.isLt⟩
  · intro i _ j _ h
    exact Fin.ext (congrArg (fun z : Fin B => z.val) h)
  · intro j hj
    exact ⟨⟨j.val,(Finset.mem_filter.mp hj).2⟩,Finset.mem_univ _,Fin.ext rfl⟩
  · intro i _
    simp only [dif_pos (lt_of_lt_of_le i.isLt cover)]
    rfl

end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.LogicalColumns
