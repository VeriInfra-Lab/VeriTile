import bench.repairs.tritonbench_g.matmul_leakyrelu.Proofs.Contract
import bench.tritonbench_g.matmul_leakyrelu.Correct

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
noncomputable section

def runChecked (roundHalf : ℝ → ℝ) (p : Metadata) (activation : String) (baseC : Nat) (initial : State ℝ) :
    Except (Sum InputError HostError) (RegionName × State ℝ) :=
  match checkInputs p with
  | .error why => .error (.inl why)
  | .ok () => if outputCheck p baseC then (Contracted.host (idealAlgebra roundHalf) Contracted.idealDotAcc
      (request p activation) initial).mapError Sum.inr else .error (.inl .outputAddress)

/-- The checked admission establishes the existing finite source envelope.
The complete mathematical host/grid theorem is reused without reproof. -/
specification checked_mathematical (roundHalf : ℝ → ℝ) (p : Metadata) (activation : String)
    (initial : State ℝ) (admitted : InputContract p)
    (A : Fin (request p activation).rows → Fin (request p activation).inner → ℝ)
    (B : Fin (request p activation).inner → Fin (request p activation).columns → ℝ)
    (hA : ∀ i k, inputA (request p activation) initial i.val k.val = A i k)
    (hB : ∀ k j, inputB (request p activation) initial k.val j.val = B k j)
    (ordinary : OrdinaryTensorContract p activation initial)
    (output : outputCheck p ordinary.baseC=Bool.true) :
    VeriTile.Spec.Real (∃ final, runChecked roundHalf p activation ordinary.baseC initial = .ok ("c", final) ∧
      (∀ i : Fin (request p activation).rows, ∀ j : Fin (request p activation).columns,
        final.mem "c" (outputOffset (request p activation) i.val j.val) =
          Cell.mk .fp16 (roundHalf (formula (request p activation) A B i j))) ∧
      (∀ region offset, ¬Footprint (request p activation) region offset →
        final.mem region offset = initial.mem region offset) ∧
      SourceSafety (request p activation) initial (ordinary.admitted admitted output)) := by
  obtain ⟨final, he, hv, hf, hs⟩ :=
    Contracted.matmul_leakyrelu_contracted_mathematical_host_correctness roundHalf
      (request p activation) initial A B hA hB (ordinary.admitted admitted output) (admitted_indices p admitted activation)
  refine ⟨final, ?_, hv, hf, hs⟩
  simp only [runChecked, (checkInputs_iff p).2 admitted,output,↓reduceIte]
  rw [he]
  rfl

#axiomsClean checked_mathematical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.Correct
