import bench.repairs.tritonbench_g.vector_addition.Proofs.Contract
import bench.repairs.tritonbench_g.vector_addition.Proofs.FiniteSource
import bench.tritonbench_g.vector_addition.Proofs.TypedGrid

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.FPProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def parameters (n : Nat) : ValueProofs.Parameters := ⟨"x", "y", "out", n, 1024⟩

/-- Formula-only adapter for the existing scatter and grid memory lemmas.
Execution below always uses M directly on the canonical explicit FP32 source.
This adapter records both FP32 load boundaries and explicit add precision. -/
def referenceAlgebra {α : Type} (M : Algebra α) : Algebra α :=
  { M with binary := fun _ d op a b =>
      M.binary (some .fp32) d op (M.fp32Load a) (M.fp32Load b) }

/-- All eight statements of the canonical DSL source execute successfully. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (s : State α) :
    {t : State α // FP.Structural.exec M (originalKernel n) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [originalKernel, add_kernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numeric, numericLt, natLt, bop, store, Region.cast]
  rfl

/-- Actual source memory effect, retaining explicit loads and compute precision. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (s : State α) :
    (execution M n s).val.mem =
      ValueProofs.expectedMemory (referenceAlgebra M) (parameters n) s := by
  simp [execution, blockSize, ValueProofs.expectedMemory, ValueProofs.offset,
    ValueProofs.resultCell, referenceAlgebra, parameters, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
    ValueProofs.masked_scatter_memory, State.setReg]
  congr 1
  funext mem i
  by_cases hi : s.pids 0 * 1024 + i.1.val < n
  · simp only [hi, ↓reduceIte]
  · simp only [hi, ↓reduceIte]

def runGridPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k+1 => (runGridPrefix M n initial k).bind fun previous =>
      FP.Structural.exec M (originalKernel n)
        (ValueProofs.programState (parameters n) initial previous.mem k)

def runGrid {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (initial : State α) :=
  runGridPrefix M n initial (ValueProofs.gridCount (parameters n))

theorem grid_prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (initial : State α) (k : Nat) :
    ∃ final, runGridPrefix M n initial k = some final ∧
      final.mem = ValueProofs.prefixMemory (referenceAlgebra M) (parameters n) initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, by funext r o; simp [ValueProofs.prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    let start := ValueProofs.programState (parameters n) initial previous.mem k
    refine ⟨(execution M n start).val, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact (execution M n start).property
    · rw [execution_memory, show start =
          ValueProofs.programState (parameters n) initial previous.mem k from rfl,
        hm, ValueProofs.reference_step]

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (initial : State α) :
    ∃ final, runGrid M n initial = some final ∧
      final.mem = ValueProofs.gridExpectedMemory (referenceAlgebra M) (parameters n) initial := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct M n initial
    (ValueProofs.gridCount (parameters n))
  exact ⟨final, he, hm.trans (ValueProofs.grid_reference_complete _ _ _ (by change 0 < (1024 : Nat); decide))⟩

/-- Logical memory uses the DSL's opaque floating carrier; FP32 words remain
unchanged as values. This is not mathematical Real memory. -/
def allocated (xs ys : Nat → Word) (scratch : State Word) : State Word :=
  {scratch with mem := fun r i =>
    if r = "x" then .mk .real (xs i)
    else if r = "y" then .mk .real (ys i)
    else scratch.mem r i}

/-- Successful-runtime execution of the shared source, with the same ten guards. -/
def runWrapper (M : Algebra Word) (x y : TensorMetadata)
    (xs ys : Nat → Word) (scratch : State Word) :
    Except InputError (Option (State Word)) := do
  checkInputs x y
  let initial := allocated xs ys scratch
  if x.numel = 0 then return some initial
  return runGrid M x.numel initial

theorem wrapper_correct (M : Algebra Word) (x y : TensorMetadata)
    (h : InputContract x y) (xs ys : Nat → Word) (scratch : State Word) :
    ∃ final, runWrapper M x y xs ys scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.mem "out" i =
        .mk .real (M.binary (some .fp32) .real .add
          (M.fp32Load (xs i)) (M.fp32Load (ys i)))) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs ys scratch).mem r i) := by
  have accepted := (check_tensor_arguments_iff x y).mpr h
  by_cases empty : x.numel = 0
  · exact ⟨allocated xs ys scratch, by simp [runWrapper, accepted, empty],
      by simp [empty], by intro r i _; rfl⟩
  · obtain ⟨final, he, hm⟩ := grid_correct M x.numel (allocated xs ys scratch)
    refine ⟨final, by simp [runWrapper, accepted, empty, he], ?_, ?_⟩
    · intro i hi
      rw [hm]
      simp [ValueProofs.gridExpectedMemory, ValueProofs.resultCell,
        referenceAlgebra, parameters, allocated, hi]
    · intro r i outside
      rw [hm]
      simp only [ValueProofs.gridExpectedMemory, parameters]
      rw [if_neg (by rcases outside with hr | hi
                     · exact fun h => hr h.1
                     · exact fun h => (Nat.not_lt.mpr hi) h.2)]

/-- FP32-word interpretation used by the physical memory model: loading an
already decoded FP32 word is identity, and the tagged addition is nativeAdd.
This defines a boundary model; it assumes no numerical identity or IEEE law. -/
def nativeAlgebra (M : Algebra Word) (nativeAdd : Word → Word → Word) : Algebra Word :=
  { M with
    fp32Load := id
    binary := fun precision dtype operation a b =>
      match precision, dtype, operation with
      | some .fp32, .real, .add => nativeAdd a b
      | _, _, _ => M.binary precision dtype operation a b }

/-- Direct canonical-source execution with the same word addition primitive
consumed by the pointer-bound finite interpreter. -/
theorem wrapper_native_correct (M : Algebra Word) (nativeAdd : Word → Word → Word)
    (x y : TensorMetadata) (h : InputContract x y)
    (xs ys : Nat → Word) (scratch : State Word) :
    ∃ final, runWrapper (nativeAlgebra M nativeAdd) x y xs ys scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.mem "out" i = .mk .real (nativeAdd (xs i) (ys i))) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs ys scratch).mem r i) := by
  simpa [nativeAlgebra] using wrapper_correct (nativeAlgebra M nativeAdd) x y h xs ys scratch

/-- Mathematical projection only; FP precision is checked separately by the
finite source lowerer and by the direct structural execution theorem above. -/
theorem source_lowering (n : Nat) :
    VeriTile.Bench.TritonBenchG.AddExample.Finite.lowerBody
      (originalKernel n).toAlgKernel.body =
        some (FiniteSource.sourceCode "x" "y" "out" n) :=
  FiniteSource.original_ast_lowering "x" "y" "out" n

/- The earlier typed-AST interface remains available for downstream clients.
It is a legacy helper, not the authoritative DSL source or a semantic bridge. -/
namespace LegacyTyped
def allocated (xs ys : Nat → Word) (scratch : State Word) : State Word :=
  {scratch with mem := fun r i =>
    if r = "x" then .mk .fp32 (xs i)
    else if r = "y" then .mk .fp32 (ys i)
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
      (parameters x.numel) (allocated xs ys scratch) (by change 0 < (1024 : Nat); decide)
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


#axiomsClean wrapper_correct
end LegacyTyped

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean grid_correct
#axiomsClean source_lowering
#axiomsClean wrapper_native_correct
#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.FPProof
