import bench.tritonbench_g.fused_activation.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.FusedActivation.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
open scoped VeriTile.Spec
noncomputable section
abbrev Rules := FP.ScalarArithmetic.Rules

def parameters (_W N B : Nat) : Parameters := ⟨"input", "bias", "x", N, B⟩

/-- Finite outer-add operands only on actually stored lanes, after the common
load prefix. This is the rewrite domain, not a Python input assertion. -/
def original (W N B : Nat) (a : Bool) : Program where
  kernel := candidateKernel (parameters W N B) W (1/2) a
  domain := [Rewrite.additionSite (parameters W N B) W (1/2) a]

def commuted (W N B : Nat) (a : Bool) : Program :=
  { original W N B a with kernel := commutedKernel "x" "bias" "input" W N B (1/2) a }

/-- Source-level FP32 equivalence by the shared strict addition commutation theorem. Both
full kernels execute successfully and all memory cells agree. Native FMA
contraction is handled separately by the native graph theorem. -/
specification fused_activation_commuted_equiv (W N B : Nat) (a : Bool) :
    original W N B a ≡[([] : Spec.Assumptions FP.GuardedFragment)] commuted W N B a := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D laws s domain
  exact Rewrite.runs_equiv M D laws (parameters W N B) W (1/2) a s
    (domain _ (by simp [original]))


#stmtSurfaceSubset fused_activation_commuted_equiv ⊆
  [original, commuted, Spec.Equiv, Spec.Assumptions, Spec.RuleEntry, FP.GuardedFragment, Program,
   FP.ProfiledMemory.instProgramSyntaxProgram]
#print_fp_assumptions fused_activation_commuted_equiv
#axiomsClean fused_activation_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedActivation.FPEquiv
