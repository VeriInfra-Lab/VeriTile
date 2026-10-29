import bench.repairs.tritonbench_g.max_reduction.IO
import bench.repairs.tritonbench_g.max_reduction.Proofs.Contract
import bench.repairs.tritonbench_g.max_reduction.Proofs.KernelMath
import bench.tritonbench_g.max_reduction.Proofs.DimensionIO

namespace VeriTile.Bench.Repairs.MaxReduction.RawReal
open VeriTile.Triton
noncomputable section

theorem stage1_source (inp mid : RegionName) (N B : Nat) :
    (PublicIO.stage1Real inp mid N B).kernel=(stage1 inp mid N B).eraseDType := rfl

theorem stage2_source (mid out : RegionName) (N B : Nat) :
    (PublicIO.stage2Real mid out N B).kernel=(stage2 mid out N B).eraseDType := rfl

/-- Each active row computes its full-axis mathematical maximum and first
maximizing index, storing FP32 values and signed-int64 indices. The public
width is the wrapper's next power of two, so no reduction-axis input is
silently omitted. Flat placements, active access bounds and memory frames are
part of DimensionIO; K>0 gives distinct output rows. -/
theorem dimension_math_correct (inp ov oi : RegionName) (M N K BM : Nat)
    (hRegions : ov ≠ oi) (hK : 0 < K) :
    Spec.Real (VeriTile.Bench.TritonBenchG.MaxReduction.DimensionIO .fp32 Bool.false
      inp ov oi M N K BM (nextPower N)) := by
  apply VeriTile.Bench.TritonBenchG.MaxReduction.dimensionIO_correct
  · exact nextPower_positive N
  · exact hRegions
  · intro i j same
    apply Fin.ext
    exact Nat.eq_of_mul_eq_mul_right hK same

/-- Independent attained-maximum/least-index theorem for every active
row of one actual kernel program. Unlike a reducer-defined output formula,
IsLeastArgmax states attainment, the upper-bound property and the least tie
index directly over the original input values. -/
theorem dimension_argmax_correct (M N K BM p k : Nat) (hN : 0 < N)
    (plan : ReductionPlan (nextPower N)) (hk : k < K) (input : Nat → ℝ)
    (s : FP.Structural.State (WithBot ℝ))
    (binding : ∀ j, j < M*N*K → (s.mem "input" j).read .real = (input j : WithBot ℝ)) :
    Spec.Real (∃ t,
      FP.Structural.exec (KernelMath.engine Compute.real plan.tree).algebra
        (dimension "input" "values" "indices" M N K BM (nextPower N))
        (SourceGrid.positioned s p k) = some t ∧
      (∀ i : Fin BM, p*BM+i.val < M → ∃ (answer : ℝ) (index : Nat),
        t.mem "values" ((p*BM+i.val)*K+k) = .mk .real (answer : WithBot ℝ) ∧
        t.mem "indices" ((p*BM+i.val)*K+k) = .mk .int (index : Int) ∧
        RealModel.IsLeastArgmax (fun n => input ((p*BM+i.val)*N*K+n*K+k)) N answer index) ∧
      (∀ (r : RegionName) (o : Nat),
        (r ≠ "values" ∧ r ≠ "indices") ∨
          (∀ i : Fin BM, p*BM+i.val < M → o ≠ (p*BM+i.val)*K+k) →
        t.mem r o = s.mem r o)) :=
  KernelMath.dimension_rows M N K BM (nextPower N) p k hN (nextPower_covers N)
    plan hk input s binding

/-- The typed source used by the dimension IO theorem is the same DSL
source used by FPEquiv, including its signed-index output stream. -/
theorem dimension_source (inp ov oi : RegionName) (M N K BM BN : Nat) :
    VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource .fp32 Bool.false
      inp ov oi M N K BM BN = dimension inp ov oi M N K BM BN := rfl

#axiomsClean dimension_math_correct
#axiomsClean dimension_argmax_correct
end
end VeriTile.Bench.Repairs.MaxReduction.RawReal
