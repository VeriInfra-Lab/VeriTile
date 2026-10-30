import bench.repairs.tritonbench_g.matrix_transpose.Proofs.KernelFP
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.MatrixTranspose.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- No floating transformation law is needed for exact index multiplication. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (m d : Nat) : Program where
  kernel := source "matrix" "out" m d
  domain := []

def indexCommuted (m d : Nat) : Program :=
  { original m d with kernel := indexCommutedSource "matrix" "out" m d }

/-- The actual existing index-commuted variant and the original both succeed
and agree on every typed memory cell. There are no numerical assumptions or
finite-value restrictions; local execution is proved rather than assumed. -/
specification transpose_index_equiv (m d : Nat) (_h : InputContract m d) :
    original m d ≡[assumptions] indexCommuted m d := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  obtain ⟨t, originalRun, candidateRun⟩ :=
    KernelFP.variants_execute (FP.ProfiledRewrite.engine M) m d s
  exact ⟨t, t, originalRun, candidateRun, rfl⟩

#print_fp_assumptions transpose_index_equiv
#axiomsClean transpose_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.FPEquiv
