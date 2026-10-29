import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.HeadlineMath
noncomputable section

/-- The retained recursive mathematical sum is the ordinary finite sum. -/
theorem rsum_eq (f : Nat → ℝ) (n : Nat) :
    Int8DequantMatmul.rsum f n = ∑ k ∈ Finset.range n, f k := by
  induction n with
  | zero => simp [Int8DequantMatmul.rsum]
  | succ n ih => simp [Int8DequantMatmul.rsum,Finset.sum_range_succ,ih]

#axiomsClean rsum_eq
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.HeadlineMath
