import bench.repairs.tritonbench_g.matmul_leakyrelu.Proofs.Contract
import bench.tritonbench_g.matmul_leakyrelu.FPEquiv

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
noncomputable section

def runChecked {α : Type} [Inhabited α] (M : Algebra α) (D : FPFused.DotAcc α)
    (p : Metadata) (activation : String) (baseC : Nat) (initial : State α) :
    Except (Sum InputError HostError) (RegionName × State α) :=
  match checkInputs p with
  | .error why => .error (.inl why)
  | .ok () => if outputCheck p baseC then (Contracted.host M D (request p activation) initial).mapError Sum.inr else .error (.inl .outputAddress)

/-- Exact observed live-accumulator dot/right-slope profile and FP16 result.
The six primitive numerical families stay explicit; source/fused floating
identity is neither assumed nor asserted. -/
specification checked_floating {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Contracted.Numerics α)
    (law : Contracted.PrimitiveLaws M D Q) (p : Metadata) (activation : String)
    (initial : State α) (admitted : InputContract p)
    (A : Fin (request p activation).rows → Fin (request p activation).inner → α)
    (B : Fin (request p activation).inner → Fin (request p activation).columns → α)
    (hA : ∀ i k, inputA (request p activation) initial i.val k.val = A i k)
    (hB : ∀ k j, inputB (request p activation) initial k.val j.val = B k j)
    (ordinary : OrdinaryTensorContract p activation initial)
    (output : outputCheck p ordinary.baseC=Bool.true) :
    ∃ final, runChecked M D p activation ordinary.baseC initial = .ok ("c", final) ∧
      (∀ i : Fin (request p activation).rows, ∀ j : Fin (request p activation).columns,
        final.mem "c" (outputOffset (request p activation) i.val j.val) =
          Cell.mk .fp16 (Contracted.logicalReference Q (request p activation) A B i j)) ∧
      (∀ region offset, ¬Footprint (request p activation) region offset →
        final.mem region offset = initial.mem region offset) ∧
      SourceSafety (request p activation) initial (ordinary.admitted admitted output) := by
  obtain ⟨final, he, hv, hf, hs⟩ :=
    Contracted.matmul_leakyrelu_contracted_fp32_host_correctness M D Q law
      (request p activation) initial A B hA hB (ordinary.admitted admitted output) (admitted_indices p admitted activation)
  refine ⟨final, ?_, hv, hf, hs⟩
  simp only [runChecked, (checkInputs_iff p).2 admitted,output,↓reduceIte]
  rw [he]
  rfl

#axiomsClean checked_floating
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.FPEquiv
