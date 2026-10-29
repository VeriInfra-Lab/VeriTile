import bench.repairs.tritonbench_g.max_reduction.IO
import bench.repairs.tritonbench_g.max_reduction.Proofs.AlgorithmSource
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
theorem legacy_dimension_math_correct (inp ov oi : RegionName) (M N K BM : Nat)
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

/-- Complete flattened execution has the same success/failure and final
memory as the prior mathematical source. Integer register carriers differ,
so this is an evaluated memory connection, not a source AST equality. -/
theorem dimension_source (A : FlatAlloc) (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (s : BlockState) :
    (exec (A.flattenKernel (dimension inp ov oi M N K BM BN).eraseDType.toAlgKernel) s).map BlockState.mem =
    (exec (A.flattenKernel (VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource .fp32 Bool.false
      inp ov oi M N K BM BN).eraseDType.toAlgKernel) s).map BlockState.mem := by
  rw [VeriTile.Bench.TritonBenchG.MaxReduction.dimension_flat_exec]
  exact AlgorithmSource.dimension_memory A inp ov oi M N K BM BN hN s

/-- The retained complete flat-placement mathematical scope now names the
actual raw repaired source, including both value/index outputs and every
exterior memory cell. -/
theorem dimension_math_correct (inp ov oi : RegionName) (M N K BM : Nat)
    (hRegions : ov ≠ oi) (hK : 0 < K) :
    Spec.Real (PublicIO.flatDimensionReal inp ov oi M N K BM (nextPower N)) := by
  intro A hd hr hc s xs hu hi hv hx hpins
  obtain ⟨t,he,values,frame⟩ := legacy_dimension_math_correct inp ov oi M N K BM hRegions hK
    A hd hr hc s xs hu hi hv hx hpins
  have same := dimension_source A inp ov oi M N K BM (nextPower N)
    (nextPower_positive N) (A.flattenState s)
  rw [he] at same
  cases actual : exec (A.flattenKernel
      (dimension inp ov oi M N K BM (nextPower N)).eraseDType.toAlgKernel) (A.flattenState s) with
  | none => simp [actual] at same
  | some u =>
    have memory : u.mem = t.mem := by simpa only [actual,Option.map_some,Option.some.injEq] using same
    refine ⟨u,rfl,?_,?_⟩
    · intro i active
      simpa only [BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped,memory] using values i active
    · intro r o outside
      change u.mem r o = (A.flattenState s).mem r o
      rw [memory]
      exact frame r o outside

#axiomsClean dimension_source
#axiomsClean legacy_dimension_math_correct

#axiomsClean dimension_math_correct
#axiomsClean dimension_argmax_correct
end
end VeriTile.Bench.Repairs.MaxReduction.RawReal
