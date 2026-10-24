import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Contract
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Schedule

namespace VeriTile.Bench.Repairs.MaskedAddCuda.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (gridCount)
noncomputable section

def pars (g : TensorMetadata) : Parameters := parameters "grad" "data" "mask" g.numel

/-- Caller memory is the initial memory, including the old grad. There is no
allocation, snapshot copy or copyback. Unit is the returned Python None. -/
def runWrapper {α : Type} [Inhabited α] (M : Algebra α)
    (g p m : TensorMetadata) (a : AlphaArgument) (token : ℝ)
    (initial : State α) (jobs : List Nat) : Except InputError (Option (Unit × State α)) := do
  checkInputs g p m a
  if g.numel = 0 then return some ((), initial)
  return (runSchedule M (selectedMask m) (pars g) token initial jobs).map ((), ·)

theorem wrapper_correct {α : Type} [Inhabited α] (M : Algebra α)
    (g p m : TensorMetadata) (a : AlphaArgument) (h : InputContract g p m a)
    (token : ℝ) (initial : State α) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (gridCount (pars g)))) :
    ∃ final, runWrapper M g p m a token initial jobs = .ok (some ((), final)) ∧
      final.mem = gridExpectedMemory M (selectedMask m) (pars g) token initial := by
  have accepted := (checkTensorInputs_iff g p m a).mpr h
  by_cases empty : g.numel = 0
  · refine ⟨initial, by simp [runWrapper, checkInputs, accepted, empty], ?_⟩
    funext r i
    simp [gridExpectedMemory, pars, parameters, empty]
  · obtain ⟨final, he, hm⟩ := complete_schedule_correct M (selectedMask m) (pars g)
      token initial (by change 0 < (1024 : Nat); decide) jobs schedule
    exact ⟨final, by simp [runWrapper, checkInputs, accepted, empty, he], hm⟩

theorem wrapper_frame {α : Type} [Inhabited α] (M : Algebra α)
    (g p m : TensorMetadata) (a : AlphaArgument) (h : InputContract g p m a)
    (token : ℝ) (initial : State α) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (gridCount (pars g)))) :
    ∃ final, runWrapper M g p m a token initial jobs = .ok (some ((), final)) ∧
      (∀ r i, (r ≠ "grad" ∨ g.numel ≤ i ∨ (initial.mem "mask" i).read .int ≠ 0) →
        final.mem r i = initial.mem r i) ∧
      final.mem = gridExpectedMemory M (selectedMask m) (pars g) token initial := by
  obtain ⟨final, he, hm⟩ := wrapper_correct M g p m a h token initial jobs schedule
  refine ⟨final, he, ?_, hm⟩
  intro r i outside
  rw [hm]
  rcases outside with hr | hi | hz
  · simp [gridExpectedMemory, pars, parameters, hr]
  · simp [gridExpectedMemory, pars, parameters, Nat.not_lt.mpr hi]
  · simp [gridExpectedMemory, pars, parameters, hz]

#axiomsClean wrapper_correct
#axiomsClean wrapper_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.Host
