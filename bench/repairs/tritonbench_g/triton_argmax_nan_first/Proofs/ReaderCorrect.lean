import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.ReaderCorrect
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

theorem firstMaximum_eq (n : Nat) (positive : 0 < n) (xs : Fin n → ℝ) :
    Classical.choose (show ∃ i : Fin n,
      (∀ j, xs j ≤ xs i) ∧ (∀ j, xs j = xs i → i.val ≤ j.val) from
      ⟨Math.argmax n positive xs, Math.argmax_spec n positive xs⟩) =
      Math.argmax n positive xs :=
  Reduction.least_maximum_unique xs _ _
    (Classical.choose_spec (show ∃ i : Fin n,
      (∀ j, xs j ≤ xs i) ∧ (∀ j, xs j = xs i → i.val ≤ j.val) from
      ⟨Math.argmax n positive xs, Math.argmax_spec n positive xs⟩))
    (Math.argmax_spec n positive xs)

theorem global (kind : Request.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (inputValue : X ≠ V) (inputIndex : X ≠ Region.cast I)
    (scratchApart : V ≠ Region.cast I) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (index64 : Bool) (select : (Fin c.size → ℝ) → Fin c.size)
    (same : ∀ xs, select xs = Math.argmax c.size coverage.sizePositive xs) :
    (IO.logicalGlobal (IO.projectedKind kind) rest X V I O c index64).Implements
      (fun xs => Int.ofNat (select xs).val) := by
  apply (IO.global_iff (IO.projectedKind kind) rest X V I O c index64 _).2
  simpa only [same, RealIO.mathematical] using
    RealIO.implements (IO.projectedKind kind) rest X V I O c coverage index64
      inputValue inputIndex scratchApart

theorem dimension (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) (positive : 0 < BN)
    (nonempty : 0 < N) (innerPositive : 0 < K) (select : (Fin N → ℝ) → Fin N)
    (same : ∀ xs, select xs = Math.argmax N nonempty xs) :
    (IO.logicalDimension rest X O M N K BM BN base pid index64).Implements
      (fun xs j => Int.ofNat (select (IO.dimensionRow M N K BM base pid xs j)).val) := by
  apply (IO.dimension_iff rest X O M N K BM BN base pid index64
    (fun row => Int.ofNat (select row).val)).2
  simpa only [same, RealDimensionIO.mathematical] using
    RealDimensionIO.implements rest X O M N K BM BN base pid index64 positive nonempty innerPositive

#axiomsClean firstMaximum_eq
#axiomsClean global
#axiomsClean dimension
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.ReaderCorrect
