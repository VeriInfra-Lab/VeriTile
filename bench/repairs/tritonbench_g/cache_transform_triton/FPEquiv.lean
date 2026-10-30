import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.PairIO

/-! Exact whole-grid equivalence of the original byte gather and its quotient
remainder address optimization. No floating operation changes, and no numerical
assumption or experiment is needed. The public exact-cell input mode preserves
the accepted metadata, byte payload and exterior-memory scope. -/
namespace VeriTile.Bench.Repairs.CacheTransform.Equivalence
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original {A : Type} (M : Algebra A) (p : Parameters) (n capacity : Nat) :=
  ProgramIO.Implementation.original (cacheIO M p n capacity) (CacheIO.Domain M p n capacity) .exactCells
abbrev quotientRemainder {A : Type} (M : Algebra A) (p : Parameters) (n capacity : Nat) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate M p n capacity) (CacheIO.Domain M p n capacity) .exactCells

/-- The complete actual source/candidate grids, all output bytes and both
source safety/exterior-frame results on the original accepted domain. -/
specification quotient_remainder_equiv {A : Type} [Inhabited A]
    (M : Algebra A) (p : Parameters) (n capacity : Nat)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit) :
    original M p n capacity ≡[[]] quotientRemainder M p n capacity := by
  exact ProgramIO.Implementation.of_equivalentOnCells (PairIO.candidate M p n capacity)
    (PairIO.equivalent M p n capacity start block grid)

#axiomsClean quotient_remainder_equiv
end
end VeriTile.Bench.Repairs.CacheTransform.Equivalence
