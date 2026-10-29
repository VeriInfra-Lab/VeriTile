import bench.repairs.tritonbench_g.add_example.Proofs.Contract
import bench.tritonbench_g.vector_addition.Proofs.TypedGrid

namespace VeriTile.Bench.Repairs.AddExample.Contiguous.FPProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Adapter parameters for reusing the existing addition proof. -/
def parameters (n : Nat) : ValueProofs.Parameters := ⟨"x", "y", "out", n, 4⟩

/-- Historical typed-grid model retained for optional wrapper helpers.
Current public proofs execute the canonical DSL source directly. -/
def legacyKernel (n : Nat) : ComputeKernel :=
  TypedFloat.materialized .fp32 .fp32 .fp32 (parameters n)

/-- The historical model is the existing typed-grid instance. -/
theorem source_for_existing_proof (n : Nat) :
    legacyKernel n = TypedFloat.materialized .fp32 .fp32 .fp32 (parameters n) := rfl

/-- The historical typed-grid model has the same mathematical projection. -/
theorem same_source (n : Nat) :
    (legacyKernel n).eraseDType.toAlgKernel.body =
      (add_kernel "x" "y" "out" n 4).toAlgKernel.body := by
  exact TypedFloat.source_projection .fp32 .fp32 (parameters n)

def allocated (xs ys : Nat → Word) (scratch : State Word) : State Word :=
  {scratch with mem := fun r i =>
    if r = "x" then .mk .fp32 (xs i)
    else if r = "y" then .mk .fp32 (ys i)
    else if r = "out" then .mk .fp32 (FP.Value.zero .fp32)
    else scratch.mem r i}

/-- Successful-runtime structural interpretation, with the same ten guards. -/
def runWrapper (M : Algebra Word) (x y : TensorMetadata)
    (xs ys : Nat → Word) (scratch : State Word) :
    Except InputError (Option (State Word)) := do
  checkInputs x y
  let initial := allocated xs ys scratch
  if x.numel = 0 then return some initial
  return TypedFloat.runGrid M .fp32 .fp32 .fp32 (parameters x.numel) initial

theorem wrapper_correct (M : Algebra Word) (x y : TensorMetadata)
    (h : InputContract x y) (xs ys : Nat → Word) (scratch : State Word) :
    ∃ final, runWrapper M x y xs ys scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.mem "out" i =
        .mk .fp32 (M.binary none .fp32 .add (xs i) (ys i))) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs ys scratch).mem r i) := by
  have accepted := (check_tensor_arguments_iff x y).mpr h
  by_cases empty : x.numel = 0
  · exact ⟨allocated xs ys scratch, by simp [runWrapper, accepted, empty],
      by simp [empty], by intro r i _; rfl⟩
  · obtain ⟨final, he, hm⟩ := TypedFloat.grid_correct M .fp32 .fp32 .fp32
      (parameters x.numel) (allocated xs ys scratch) (by change 0 < (4 : Nat); decide)
    refine ⟨final, by simp [runWrapper, accepted, empty, he], ?_, ?_⟩
    · intro i hi
      rw [hm]
      simp [TypedFloat.gridExpectedMemory, TypedFloat.resultCell,
        TypedFloat.castValue, parameters, allocated, hi, ofFloat, toFloat]
    · intro r i outside
      rw [hm]
      simp only [TypedFloat.gridExpectedMemory, parameters]
      rw [if_neg (by rcases outside with hr | hi
                     · exact fun h => hr h.1
                     · exact fun h => (Nat.not_lt.mpr hi) h.2)]

#axiomsClean source_for_existing_proof
#axiomsClean same_source
#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddExample.Contiguous.FPProof
