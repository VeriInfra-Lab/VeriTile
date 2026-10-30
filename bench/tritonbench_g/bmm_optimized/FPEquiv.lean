import bench.tritonbench_g.bmm_optimized.Proofs.PairIO

/-! Full original/integer-CSE program equivalence under the retained live-FMA
interpretation. The candidate exposes a shared quotient in group remainder
calculation; all floating operations and native words are unchanged. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Equivalence
open VeriTile Triton FP.Structural BmmOptimized
open scoped VeriTile.Spec
noncomputable section

abbrev original {A : Type} (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) :=
  ProgramIO.Implementation.original (Interface.io s c p rest ps) (fun _ => True)

abbrev quotientRemainder {A : Type} (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate s c p rest ps) (fun _ => True)

/-- Exact complete launch equivalence. No numerical rewrite, finite guard or
experimental rule is needed; the actual FMA order and output observer are shared. -/
specification quotient_remainder_equiv {A : Type} [Inhabited A]
    (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) (order : ps.Perm (programs s c)) :
    original s c p rest ps ≡[[]] quotientRemainder s c p rest ps := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.candidate s c p rest ps)
    (PairIO.equivalent s c p rest ps order)

#axiomsClean quotient_remainder_equiv
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Equivalence
