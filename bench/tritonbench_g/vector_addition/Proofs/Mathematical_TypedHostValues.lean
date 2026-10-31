import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedHostWrapper

namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
open VeriTile.Triton FP.Structural DTypeScope
open ValueProofs (Parameters)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def floatWordWidth : TypedFloat.Storage → Nat
  | .fp16 | .bf16 => 16 | .fp32 => 32 | .f8e4 | .f8e5 => 8

/-- Concrete tag and finite-word input relation for the shared carrier.
Small floats occupy low bits, as in the inherited mixed-small interpreter. -/
def SharedScalarView (d : DType) (cell : Cell Word) : Prop :=
  match integerStorage d with
  | some k => ∃ v : IntegerWords.Word k, cell = IntegerWords.decodeCell k v
  | none => match floatStorage d with
    | some f => ∃ v : Word, cell = .mk f.dtype.toTileDType (ofFloat f.dtype v) ∧ v.bits.toNat < 2^(floatWordWidth f)
    | none => False

def InputViews {σ : Type} (r : Route) (p : Parameters) (s : World σ) : Prop :=
  match r with
  | .wide x y => ∃ (xs : Nat → FP64TypedCore.Word (.storage x)) (ys : Nat → FP64TypedCore.Word (.storage y)),
      FP64Memory.InputWords (wideParameters x y p) xs ys s.native.wide
  | _ => ∀ i, i < p.n → SharedScalarView r.signature.1 (s.kernel.mem p.x i) ∧
      SharedScalarView r.signature.2 (s.kernel.mem p.y i)

/-- Defined conversion domain for source-corresponding results. Outside this
domain the shared interpreters retain their explicit totalization policies;
wide interpreter diagnostics are not treated as native exceptions. -/
def DefinedConversions {σ : Type} (b : Numerics) (r : Route) (p : Parameters) (s : World σ) : Prop :=
  match r with
  | .mixed32 first d => MixedFP32.OutputDefined b.cfg first d.storage p s.kernel
  | .mixedSmall f first d => MixedSmall.OutputDefined b.cfg f first d.storage p s.kernel
  | _ => True

def ValueDomain {σ : Type} (b : Numerics) (r : Route) (p : Parameters) (s : World σ) : Prop :=
  Ready b r p s ∧ InputViews r p s ∧ DefinedConversions b r p s

def definedExpected {σ : Type} (b : Numerics) (r : Route) (p : Parameters) (s : World σ) : ExternalMemory :=
  match r with
  | .mixed32 first d => .shared (MixedFP32.specifiedMemory b.cfg first d.storage p s.kernel)
  | .mixedSmall f first d => .shared (MixedSmall.specifiedMemory b.cfg f first d.storage p s.kernel)
  | _ => expected b r p s

theorem expected_policy_independent {σ : Type} (b : Numerics) (r : Route)
    (p : Parameters) (s : World σ) (defined : DefinedConversions b r p s) :
    expected b r p s = definedExpected b r p s := by
  cases r with
  | integer _ _ => rfl
  | floating _ _ => rfl
  | wide _ _ => rfl
  | mixed32 first d =>
    exact congrArg ExternalMemory.shared
      (MixedFP32.grid_memory_policy_independent b.cfg b.undefinedSigned (fun _ _ => 0)
        b.undefinedUnsigned (fun _ _ => 0) first d.storage p s.kernel defined)
  | mixedSmall f first d =>
    exact congrArg ExternalMemory.shared
      (MixedSmall.grid_memory_policy_independent b.cfg f b.undefinedSigned (fun _ _ => 0)
        b.undefinedUnsigned (fun _ _ => 0) first d.storage p s.kernel defined)

theorem integer_expected_words {σ : Type} (b : Numerics) (x y : IntegerSource.Canonical)
    (wx : Nat → IntegerWords.Word x) (wy : Nat → IntegerWords.Word y)
    (p : Parameters) (s : World σ)
    (hx : IntegerWords.InputWords x wx p.x p.n s.kernel.mem)
    (hy : IntegerWords.InputWords y wy p.y p.n s.kernel.mem) :
    expected b (.integer x y) p s = .shared (IntegerWords.wordSpec x y wx wy p s.kernel.mem) :=
  congrArg ExternalMemory.shared (IntegerWords.reference_refines x y wx wy p s.kernel hx hy)

def wideIntegerOutput : Route → Bool
  | .wide x _ => !x.isFloating
  | _ => Bool.false

def wideFloatingOutput : Route → Bool
  | .wide x _ => x.isFloating
  | _ => Bool.false

theorem wide_output_domain_inventory :
    (routes.filter wideIntegerOutput).length = 9 ∧
    (routes.filter wideFloatingOutput).length = 20 := by decide +kernel

#specNonCircular definedExpected avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean expected_policy_independent
#axiomsClean integer_expected_words
#axiomsClean wide_output_domain_inventory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
