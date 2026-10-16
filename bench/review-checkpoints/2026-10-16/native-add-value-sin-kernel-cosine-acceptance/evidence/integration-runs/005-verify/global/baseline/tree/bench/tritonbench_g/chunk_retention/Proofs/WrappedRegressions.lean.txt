import bench.tritonbench_g.chunk_retention.Proofs.IndexEvaluation
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.Decay
open VeriTile.Triton
noncomputable section

/-- The historical natural-index causal formula is false for the complete
accepted source domain. This small scalar witness does not allocate a huge
tile or claim that such a tile is a legal GPU launch. -/
theorem historical_causal_formula_refuted (base : ℝ) :
    score base 2147483648 0 ≠ idealScore base 2147483648 0 := by
  rw [wrapped_causal_witness]
  simp only [idealScore,Nat.zero_le,ite_true,Nat.cast_zero,sub_zero]
  exact ne_of_lt (factor_pos base _)

/-- The narrowing is periodic outside the ordinary signed range as well;
the repaired contracts do not add an artificial block-size upper bound. -/
theorem wrap_back_to_zero : index32 4294967296 = 0 := by
  norm_num [index32]

#axiomsClean historical_causal_formula_refuted
#axiomsClean wrap_back_to_zero
#specNonCircular score avoiding [evalOp]
end
end VeriTile.Bench.TritonBenchG.ChunkRetention.Decay
