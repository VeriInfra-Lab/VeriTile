import bench.tritonbench_g.matrix_reduction.Proofs.ProgramPair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.FPEquiv
open VeriTile.Triton FP.Structural TypedSource FPCorrect
open scoped VeriTile.Spec
noncomputable section

def original {α : Type} [Inhabited α] (A : Algebra α) (storage : Storage) (M N : Nat) :=
  ProgramIO.Implementation.original (matrixReductionNativeIO A storage M N) (fun _ => True)
def affineAddress {α : Type} [Inhabited α] (A : Algebra α) (storage : Storage) (M N : Nat) :=
  ProgramIO.Implementation.ofVariant (matrixReductionCandidate A storage M N) (fun _ => True)

/-- The full-tile affine-address candidate preserves the exact native row
maximum graph of both original FP16/FP32 storage profiles, for every input word.
Both programs are safe, succeed, and preserve the complete exterior frame. -/
specification matrix_reduction_pair {α : Type} [Inhabited α] (A : Algebra α)
    (storage : Storage) (M N : Nat) (hN : 0 < N) :
    original A storage M N ≡[[]] affineAddress A storage M N :=
  ProgramIO.Implementation.of_equivalent (matrixReductionCandidate A storage M N)
    (ProgramPair.equivalent A storage M N hN)

#axiomsClean matrix_reduction_pair
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.FPEquiv
