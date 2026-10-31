import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedHostDispatch

namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Numerics where
  rest : Algebra Word
  cfg : FP.Config
  undefinedSigned : Nat → Word → Int
  undefinedUnsigned : Nat → Word → Nat
  wide : FP64TypedCore.Numerics

structure Native (σ : Type) where
  wide : FP64Memory.Memory
  environment : σ

abbrev World (σ : Type) := HostWrapper.World Word (Native σ)

inductive Diagnostic where
  | sharedInterpreter
  | wideInterpreter (fault : FP64Memory.Fault)

/-- These outcomes belong to the mathematical interpreter. In particular an
undefined fp64-to-integer conversion is not a proved native exception, rollback,
or partially committed device-memory state. -/
abbrev Attempt (σ : Type) := Except Diagnostic (World σ)

def wideParameters (x y : FP64TypedCore.Storage) (p : Parameters) : FP64Memory.Parameters x y :=
  ⟨p.n, p.x.name, p.y.name, p.output.name⟩

def finishShared {σ : Type} (s : World σ) : Option (State Word) → Attempt σ
  | none => .error .sharedInterpreter
  | some t => .ok { s with kernel := t }

inductive ExternalMemory where
  | shared (memory : RegionName → Nat → Cell Word)
  | wide (memory : FP64Memory.Memory)

/-- The exact branch-specific external relation: 160 branches observe the
tagged shared cells; 29 fp64 branches observe distinct-width typed words. -/
def project {σ : Type} (r : Route) (s : World σ) : ExternalMemory :=
  match r with
  | .wide _ _ => .wide s.native.wide
  | _ => .shared s.kernel.mem

/-- An independent initial-memory formula, with no fallback values for an
absent/wrong-tag fp64 input or an undefined conversion. -/
def wideExpected (b : FP64TypedCore.Numerics) (x y : FP64TypedCore.Storage)
    (p : Parameters) (mem : FP64Memory.Memory) : FP64Memory.Memory := fun r i =>
  if r = p.output.name ∧ i < p.n then do
    let a ← FP64Memory.readCell x (mem p.x.name i)
    let c ← FP64Memory.readCell y (mem p.y.name i)
    let v ← FP64TypedCore.result b x y a c
    pure (FP64Memory.Cell.mk x v)
  else mem r i

def expected {σ : Type} (b : Numerics) (r : Route) (p : Parameters) (s : World σ) : ExternalMemory :=
  match r with
  | .integer x y => .shared (IntegerSource.gridExpectedMemory x.storage y.storage p s.kernel)
  | .floating x y => .shared (TypedValueCorrect.spec b.rest x y p s.kernel)
  | .mixed32 first d => .shared
      (MixedFP32.gridExpectedMemory b.cfg b.undefinedSigned b.undefinedUnsigned first d.storage p s.kernel)
  | .mixedSmall f first d => .shared
      (MixedSmall.gridExpectedMemory b.cfg f b.undefinedSigned b.undefinedUnsigned first d.storage p s.kernel)
  | .wide x y => .wide (wideExpected b.wide x y p s.native.wide)

/-- The 160 shared routes use their existing totalized semantics. Their native
numerical validity still requires the recorded policies/defined-value bindings.
The 29 wide routes require typed active inputs, separation, and OutputDefined;
the latter is automatic for the 20 floating-output routes, not the other nine. -/
def Ready {σ : Type} (b : Numerics) (r : Route) (p : Parameters) (s : World σ) : Prop :=
  match r with
  | .wide x y =>
      FP64Memory.Separated (wideParameters x y p) ∧
      ∃ (xs : Nat → FP64TypedCore.Word (.storage x)) (ys : Nat → FP64TypedCore.Word (.storage y)),
        FP64Memory.InputWords (wideParameters x y p) xs ys s.native.wide ∧
        FP64Memory.OutputDefined b.wide (wideParameters x y p) xs ys
  | _ => True

def Frame {σ : Type} (p : Parameters) (s t : World σ) : Prop :=
  (∀ r i, r ≠ p.output ∨ p.n ≤ i → t.kernel.mem r i = s.kernel.mem r i) ∧
  (∀ (r : String) i, r ≠ p.output.name ∨ p.n ≤ i → t.native.wide r i = s.native.wide r i) ∧
  t.native.environment = s.native.environment

def UnselectedUnchanged {σ : Type} (r : Route) (s t : World σ) : Prop :=
  match r with
  | .wide _ _ => t.kernel = s.kernel
  | _ => t.native = s.native

theorem wideExpected_reference (b : FP64TypedCore.Numerics) (x y : FP64TypedCore.Storage)
    (p : Parameters) (mem : FP64Memory.Memory)
    (xs : Nat → FP64TypedCore.Word (.storage x)) (ys : Nat → FP64TypedCore.Word (.storage y))
    (inputs : FP64Memory.InputWords (wideParameters x y p) xs ys mem) :
    wideExpected b x y p mem = FP64Memory.reference b (wideParameters x y p) xs ys mem := by
  funext r i
  by_cases h : r = p.output.name ∧ i < p.n
  · have hx : mem p.x.name i = some (.mk x (xs i)) := (inputs i h.2).1
    have hy : mem p.y.name i = some (.mk y (ys i)) := (inputs i h.2).2
    simp [wideExpected, FP64Memory.reference, wideParameters, h, hx, hy,
      FP64Memory.outputCell, Option.map_eq_bind]
  · simp [wideExpected, FP64Memory.reference, wideParameters, h]

theorem expected_shared_frame {σ : Type} (b : Numerics) (r : Route) (p : Parameters)
    (s : World σ) (mem : RegionName → Nat → Cell Word)
    (formula : expected b r p s = .shared mem) (region : RegionName) (i : Nat)
    (outside : region ≠ p.output ∨ p.n ≤ i) : mem region i = s.kernel.mem region i := by
  have inactive : ¬(region = p.output ∧ i < p.n) := by
    rintro ⟨hr, hi⟩
    rcases outside with h | h
    · exact h hr
    · omega
  cases r <;> simp only [expected, ExternalMemory.shared.injEq, reduceCtorEq] at formula
  all_goals subst mem
  all_goals simp [IntegerSource.gridExpectedMemory, TypedValueCorrect.spec,
    MixedFP32.gridExpectedMemory, MixedSmall.gridExpectedMemory, inactive]

theorem wideExpected_frame (b : FP64TypedCore.Numerics) (x y : FP64TypedCore.Storage)
    (p : Parameters) (mem : FP64Memory.Memory) (r : String) (i : Nat)
    (outside : r ≠ p.output.name ∨ p.n ≤ i) : wideExpected b x y p mem r i = mem r i := by
  simp only [wideExpected]
  rw [if_neg]
  rintro ⟨hr, hi⟩
  rcases outside with h | h
  · exact h hr
  · omega

theorem floating_wide_ready {σ : Type} (b : Numerics) (x y : FP64TypedCore.Storage)
    (p : Parameters) (s : World σ)
    (floating : x.isFloating = Bool.true)
    (separated : FP64Memory.Separated (wideParameters x y p))
    (xs : Nat → FP64TypedCore.Word (.storage x)) (ys : Nat → FP64TypedCore.Word (.storage y))
    (inputs : FP64Memory.InputWords (wideParameters x y p) xs ys s.native.wide) :
    Ready b (.wide x y) p s :=
  ⟨separated, xs, ys, inputs, FP64Memory.floating_defined b.wide _ xs ys floating⟩

#specNonCircular expected avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean wideExpected_reference
#axiomsClean expected_shared_frame
#axiomsClean wideExpected_frame
#axiomsClean floating_wide_ready
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
