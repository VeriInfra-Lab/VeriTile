import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedFP32Grid

/-! Defined-domain independence of arbitrary conversion policies. Only active
integer-output addresses are constrained. No condition is imposed on masked
lanes, fp32-output exceptional values or the source wrapper's inputs. These
are theorem domains, not invented source runtime guards. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
open VeriTile.Triton FP.Structural FP.IntegerConversion
open ValueProofs (Parameters)
open IntegerSource (Storage)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def ConversionDefined (d : Storage) (v : Word) : Prop :=
  match d with
  | .boolean => ∃ z, toSigned 8 v = some z
  | .signed width => ∃ z, toSigned width v = some z
  | .unsigned width => ∃ z, toUnsigned width v = some z

def OutputDefined (cfg : FP.Config) (integerFirst : Bool) (d : Storage)
    (p : Parameters) (initial : State Word) : Prop :=
  ∀ o, o < p.n → integerFirst = Bool.true →
    ConversionDefined d (sumValue cfg integerFirst d p initial.mem o)

theorem convertedCell_policy_independent
    (us₁ us₂ : Nat → Word → Int) (uu₁ uu₂ : Nat → Word → Nat)
    (d : Storage) (v : Word) (defined : ConversionDefined d v) :
    convertedCell us₁ uu₁ d v = convertedCell us₂ uu₂ d v := by
  cases d
  all_goals obtain ⟨z, hz⟩ := defined
  all_goals simp [convertedCell, totalSigned, totalUnsigned, hz]

theorem resultCell_policy_independent (cfg : FP.Config)
    (us₁ us₂ : Nat → Word → Int) (uu₁ uu₂ : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters)
    (mem : RegionName → Nat → Cell Word) (o : Nat)
    (defined : integerFirst = Bool.true → ConversionDefined d (sumValue cfg integerFirst d p mem o)) :
    resultCell cfg us₁ uu₁ integerFirst d p mem o =
      resultCell cfg us₂ uu₂ integerFirst d p mem o := by
  cases integerFirst
  · rfl
  · exact convertedCell_policy_independent us₁ us₂ uu₁ uu₂ d _ (defined rfl)

theorem grid_memory_policy_independent (cfg : FP.Config)
    (us₁ us₂ : Nat → Word → Int) (uu₁ uu₂ : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word)
    (defined : OutputDefined cfg integerFirst d p initial) :
    gridExpectedMemory cfg us₁ uu₁ integerFirst d p initial =
      gridExpectedMemory cfg us₂ uu₂ integerFirst d p initial := by
  funext r o
  by_cases active : r = p.output ∧ o < p.n
  · simp only [gridExpectedMemory, active, ↓reduceIte]
    exact resultCell_policy_independent cfg us₁ us₂ uu₁ uu₂ integerFirst d p initial.mem o
      (defined o active.2)
  · simp only [gridExpectedMemory, active, ↓reduceIte]

/-- Zero is merely a syntactic fallback in an independent reference. The
defined-domain theorem proves that fallback cannot affect any stored value. -/
def specifiedMemory (cfg : FP.Config) (integerFirst : Bool) (d : Storage)
    (p : Parameters) (initial : State Word) : RegionName → Nat → Cell Word :=
  gridExpectedMemory cfg (fun _ _ => 0) (fun _ _ => 0) integerFirst d p initial

theorem float_output_always_defined (cfg : FP.Config) (d : Storage)
    (p : Parameters) (initial : State Word) : OutputDefined cfg Bool.false d p initial := by
  intro o active impossible
  cases impossible

#specNonCircular OutputDefined avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#specNonCircular specifiedMemory avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean convertedCell_policy_independent
#axiomsClean resultCell_policy_independent
#axiomsClean grid_memory_policy_independent
#axiomsClean float_output_always_defined
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
