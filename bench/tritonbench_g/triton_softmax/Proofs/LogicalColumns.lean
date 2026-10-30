import bench.tritonbench_g.triton_softmax.Proofs.RealIOImplements

namespace VeriTile.Bench.Repairs.TritonSoftmax.LogicalColumns
open VeriTile.Triton
noncomputable section

/-- The masked block sum is exactly the full logical-column sum when the
public block coverage condition holds. -/
theorem sum_eq (n B : Nat) (cover : n ≤ B) (f : Fin B → ℝ) :
    (∑ j : Fin B, if j.val < n then f j else 0) =
      ∑ j : Fin n, f (Fin.castLE cover j) := by
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
    rfl

end
end VeriTile.Bench.Repairs.TritonSoftmax.LogicalColumns
