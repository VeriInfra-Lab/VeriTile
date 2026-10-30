import bench.tritonbench_g.vector_addition.Proofs.MixedSmallGrid

/-! Defined-domain independence of arbitrary conversion policies. Only active
integer-output addresses are constrained. No condition is imposed on masked
lanes, floating-output exceptional values or the source wrapper's inputs. These
are theorem domains, not invented source runtime guards. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
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

def OutputDefined (cfg : FP.Config) (f : FloatStorage) (integerFirst : Bool) (d : Storage)
    (p : Parameters) (initial : State Word) : Prop :=
  ∀ o, o < p.n → integerFirst = Bool.true →
    ConversionDefined d (conversionInput f (sumValue cfg f integerFirst d p initial.mem o))

theorem convertedCell_policy_independent
    (us₁ us₂ : Nat → Word → Int) (uu₁ uu₂ : Nat → Word → Nat)
    (d : Storage) (v : Word) (defined : ConversionDefined d v) :
    convertedCell us₁ uu₁ d v = convertedCell us₂ uu₂ d v := by
  cases d
  all_goals obtain ⟨z, hz⟩ := defined
  all_goals simp [convertedCell, MixedFP32.convertedCell, totalSigned, totalUnsigned, hz]

theorem resultCell_policy_independent (cfg : FP.Config) (f : FloatStorage)
    (us₁ us₂ : Nat → Word → Int) (uu₁ uu₂ : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters)
    (mem : RegionName → Nat → Cell Word) (o : Nat)
    (defined : integerFirst = Bool.true → ConversionDefined d (conversionInput f (sumValue cfg f integerFirst d p mem o))) :
    resultCell cfg f us₁ uu₁ integerFirst d p mem o =
      resultCell cfg f us₂ uu₂ integerFirst d p mem o := by
  cases integerFirst
  · rfl
  · exact convertedCell_policy_independent us₁ us₂ uu₁ uu₂ d _ (defined rfl)

theorem grid_memory_policy_independent (cfg : FP.Config) (f : FloatStorage)
    (us₁ us₂ : Nat → Word → Int) (uu₁ uu₂ : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word)
    (defined : OutputDefined cfg f integerFirst d p initial) :
    gridExpectedMemory cfg f us₁ uu₁ integerFirst d p initial =
      gridExpectedMemory cfg f us₂ uu₂ integerFirst d p initial := by
  funext r o
  by_cases active : r = p.output ∧ o < p.n
  · simp only [gridExpectedMemory, if_pos active]
    exact resultCell_policy_independent cfg f us₁ us₂ uu₁ uu₂ integerFirst d p initial.mem o
      (defined o active.2)
  · simp only [gridExpectedMemory, if_neg active]

/-- Zero is merely a syntactic fallback in an independent reference. The
defined-domain theorem proves that fallback cannot affect any stored value. -/
def specifiedMemory (cfg : FP.Config) (f : FloatStorage) (integerFirst : Bool) (d : Storage)
    (p : Parameters) (initial : State Word) : RegionName → Nat → Cell Word :=
  gridExpectedMemory cfg f (fun _ _ => 0) (fun _ _ => 0) integerFirst d p initial

theorem defined_grid_correct (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word)
    (positive : 0 < p.block) (defined : OutputDefined cfg f integerFirst d p initial) :
    ∃ final, runGrid rest cfg f us uu integerFirst d p initial = some final ∧
      final.mem = specifiedMemory cfg f integerFirst d p initial := by
  obtain ⟨final, he, hm⟩ := grid_correct rest cfg f us uu integerFirst d p initial positive
  refine ⟨final, he, hm.trans ?_⟩
  exact grid_memory_policy_independent cfg f us (fun _ _ => 0) uu (fun _ _ => 0)
    integerFirst d p initial defined

theorem float_output_always_defined (cfg : FP.Config) (f : FloatStorage) (d : Storage)
    (p : Parameters) (initial : State Word) : OutputDefined cfg f Bool.false d p initial := by
  intro o active impossible
  cases impossible

#specNonCircular OutputDefined avoiding [source, FP.Structural.exec, runGrid]
#specNonCircular specifiedMemory avoiding [source, FP.Structural.exec, runGrid]
#axiomsClean convertedCell_policy_independent
#axiomsClean resultCell_policy_independent
#axiomsClean grid_memory_policy_independent
#axiomsClean defined_grid_correct
#axiomsClean float_output_always_defined
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
