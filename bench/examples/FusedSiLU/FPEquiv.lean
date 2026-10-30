import bench.examples.FusedSiLU.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Complete typed fused/three-stage equivalence for every block width and all
FP32 input words, including special values, under the existing scalarFP32
software interpretation for every configuration. Sigmoid and BF16 conversion
remain the same opaque callbacks in both programs. FP32 stores/reloads preserve
words in this interpretation. The old untyped arbitrary-algebra statement is
not asserted for native typed scratch. No numerical rewrite assumption or GPU
conformance claim is introduced. Shared IO is in IO.lean: three contiguous input
tiles, one BF16 output tile, and two private FP32 scratch tiles in the candidate. -/
namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv
open VeriTile Triton FP.Structural FP.Software
attribute [local instance] pipelineWordInhabited

def fusedIO (B : Nat) : PipelineIO₃ := ⟨FusedSiLU.IO.fusedIO B⟩
def unfusedIO (B : Nat) : PipelineIO₃ := ⟨FusedSiLU.IO.unfusedIO B⟩
def R : Spec.Assumptions ComputeStmt := []

theorem word_equivalent (B : Nat) : PipelineEquivalent₃ (fusedIO B) (unfusedIO B) := by
  have h := Execution.equivalent B
  refine ⟨h.1, h.2.1, ?_⟩
  intro cfg rest s
  exact h.2.2 _ (scalarFP32 rest cfg) (fun _ => rfl) s

open scoped VeriTile.Spec
specification silu_equiv (B : Nat) : fusedIO B ≡[R] unfusedIO B := by
  apply Spec.FloatingPoint.ofStructural (lhs := fusedIO B) (rhs := unfusedIO B)
    (structural := PipelineEquivalent₃) rfl rfl
  exact word_equivalent B

#print_fp_assumptions silu_equiv
#axiomsClean word_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.FusedSiLUFPEquiv
