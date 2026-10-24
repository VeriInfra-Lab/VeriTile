import bench.repairs.tritonbench_g.fused_activation.Proofs.Contract
import bench.repairs.tritonbench_g.fused_activation.Proofs.Grid

namespace VeriTile.Bench.Repairs.FusedActivation.Host
open VeriTile.Triton FP.Structural
noncomputable section

def pars (x : TensorMetadata) : Parameters := ⟨"input", "bias", "x", x.numel, block x.numel⟩

/-- Object identity is returned unchanged. The caller's existing memory is
executed directly; device-context and successful launch are ordinary API calls. -/
def runWrapper {α : Type} [Inhabited α] (M : Algebra α) (object : Nat)
    (x b i : TensorMetadata) (initial : State α) : Except InputError (Option (Nat × State α)) := do
  checkInputs x b i
  if x.numel = 0 then return some (object, initial)
  return (runGrid M (pars x) b.numel (1/2) Bool.true initial).map (object, ·)

theorem wrapper_correct {α : Type} [Inhabited α] (M : Algebra α) (object : Nat)
    (x b i : TensorMetadata) (h : InputContract x b i) (initial : State α) :
    ∃ final, runWrapper M object x b i initial = .ok (some (object, final)) ∧
      final.mem = gridExpectedMemory M (pars x) b.numel (1/2) Bool.true initial ∧
      (∀ r j, r ≠ "x" ∨ x.numel ≤ j → final.mem r j = initial.mem r j) := by
  have accepted := (checkTensorInputs_iff x b i).mpr h
  have launch : ∃ final, runWrapper M object x b i initial = .ok (some (object, final)) ∧
      final.mem = gridExpectedMemory M (pars x) b.numel (1/2) Bool.true initial := by
    by_cases empty : x.numel = 0
    · refine ⟨initial, by simp [runWrapper,checkInputs,accepted,empty], ?_⟩
      funext r j
      simp [gridExpectedMemory,pars,empty]
    · obtain ⟨final, he, hm⟩ := grid_correct M (pars x) b.numel (1/2) Bool.true initial
        (block_positive x.numel) (by change ("bias" : RegionName) ≠ "x"; decide)
      exact ⟨final, by simp [runWrapper,checkInputs,accepted,empty]; simpa only [one_div] using he, hm⟩
  obtain ⟨final,he,hm⟩ := launch
  refine ⟨final,he,hm,?_⟩
  intro r j outside
  rw [hm]
  rcases outside with hr | hj
  · simp [gridExpectedMemory,pars,hr]
  · simp [gridExpectedMemory,pars,Nat.not_lt.mpr hj]

#axiomsClean wrapper_correct
end
end VeriTile.Bench.Repairs.FusedActivation.Host
