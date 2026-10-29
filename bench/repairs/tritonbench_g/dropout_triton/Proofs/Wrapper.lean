import bench.repairs.tritonbench_g.dropout_triton.Proofs.Contract
import bench.repairs.tritonbench_g.dropout_triton.Proofs.TypedGrid

namespace VeriTile.Bench.Repairs.DropoutTriton.Wrapper
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section

def parameters (n : Nat) : ValueProofs.Parameters := ⟨"x", "keep", "out", n, 1024⟩

theorem same_source (n : Nat) (p : ℝ) : Typed.source (parameters n) p = kernel n p := rfl

/-- Ordinary successful empty_like allocation. No field states a dropout
result. Allocation preserves the caller memory outside the fresh output. -/
structure Allocation {α : Type} (x : TensorMetadata) (before : State α) where
  /-- Opaque identity of the object returned by empty_like, separate from metadata. -/
  outputObject : Nat
  output : TensorMetadata
  state : State α
  shape : output.shape = x.shape
  device : output.device = x.device
  dtype : output.dtype = x.dtype
  contiguous : output.isContiguous = Bool.true
  noNeg : output.isNeg = Bool.false
  layout : output.layout = .strided
  alignment : 0 < x.numel → output.dataPtr.toNat % 4 = 0
  frame : ∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i → state.mem r i = before.mem r i

abbrev Allocator (α : Type) := ∀ (x : TensorMetadata) (before : State α), Allocation x before

structure Returned (α : Type) where
  outputObject : Nat
  output : TensorMetadata
  state : State α
  /-- The input-device context selects the launch device; empty calls do not launch. -/
  launchDevice : Option VeriTile.Torch.Device

def run {α : Type} [Inhabited α] (M : Algebra α) (allocate : Allocator α)
    (x keep : TensorMetadata) (p : ScalarArgument) (before : State α) :
    Except InputError (Option (Returned α)) := do
  checkInputs x keep p
  let allocation := allocate x before
  if x.numel = 0 then return some ⟨allocation.outputObject, allocation.output, allocation.state, none⟩
  match p with
  | .float (some q) =>
      return (Typed.runGrid M (parameters x.numel) q allocation.state).map
        (fun state => ⟨allocation.outputObject, allocation.output, state, some x.device⟩)
  | _ => .error .probability

/-- Full admitted wrapper, including empty allocation/return and every grid
program. This conclusion records metadata; `return_details` also records the
separate allocated-object identity and selected CUDA launch device. -/
theorem correct {α : Type} [Inhabited α] (M : Algebra α) (allocate : Allocator α)
    (x keep : TensorMetadata) (p : ScalarArgument) (h : InputContract x keep p)
    (before : State α) :
    ∃ q final, p = .float (some q) ∧
      run M allocate x keep p before = .ok (some final) ∧
      final.output = (allocate x before).output ∧
      final.output.shape = x.shape ∧ final.output.device = x.device ∧
      final.output.dtype = .floating .fp32 ∧
      (∀ i, i < x.numel → final.state.mem "out" i =
        Typed.resultCell M (parameters x.numel) q before.mem i) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i → final.state.mem r i = before.mem r i) := by
  have accepted := (checkInputs_iff x keep p).mpr h
  obtain ⟨q, hp, hq⟩ := h.scalar
  subst p
  let a := allocate x before
  have hd := a.dtype.trans h.tensors.dtype.1
  have value (i : Nat) : Typed.resultCell M (parameters x.numel) q a.state.mem i =
      Typed.resultCell M (parameters x.numel) q before.mem i := by
    simp only [Typed.resultCell, parameters]
    rw [a.frame "keep" i (Or.inl (by decide)), a.frame "x" i (Or.inl (by decide))]
  by_cases empty : x.numel = 0
  · refine ⟨q, ⟨a.outputObject, a.output, a.state, none⟩, rfl, ?_, rfl, a.shape, a.device, hd, ?_, a.frame⟩
    · simp [run, accepted, empty, a]
    · simp [empty]
  · obtain ⟨final, he, hm⟩ := Typed.grid_correct M (parameters x.numel) q a.state (by change 0 < (1024 : Nat); decide)
    refine ⟨q, ⟨a.outputObject, a.output, final, some x.device⟩, rfl, ?_, rfl, a.shape, a.device, hd, ?_, ?_⟩
    · simp [run, accepted, empty, he, a] at *
    · intro i hi
      rw [hm]
      simp only [Typed.gridExpectedMemory, parameters, hi, and_self, ↓reduceIte]
      exact value i
    · intro r i outside
      rw [hm]
      simp only [Typed.gridExpectedMemory, parameters]
      rw [if_neg (by rcases outside with hr | hi
                     · exact fun h => hr h.1
                     · exact fun h => (Nat.not_lt.mpr hi) h.2)]
      exact a.frame r i outside

theorem return_details {α : Type} [Inhabited α] (M : Algebra α) (allocate : Allocator α)
    (x keep : TensorMetadata) (p : ScalarArgument) (h : InputContract x keep p)
    (before : State α) (final : Returned α)
    (returned : run M allocate x keep p before = .ok (some final)) :
    final.outputObject = (allocate x before).outputObject ∧
    final.launchDevice = if x.numel = 0 then none else some x.device := by
  have accepted := (checkInputs_iff x keep p).mpr h
  obtain ⟨q, hp, _⟩ := h.scalar
  subst p
  by_cases empty : x.numel = 0
  · simp [run, accepted, empty] at returned
    cases returned
    simp [empty]
  · cases grid : Typed.runGrid M (parameters x.numel) q (allocate x before).state with
    | none => simp [run, accepted, empty, grid] at returned
    | some state =>
      simp [run, accepted, empty, grid] at returned
      cases returned
      simp [empty]

#axiomsClean same_source
#axiomsClean correct
#axiomsClean return_details
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Wrapper
