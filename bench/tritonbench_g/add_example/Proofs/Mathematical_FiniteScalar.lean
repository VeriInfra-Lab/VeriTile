import bench.tritonbench_g.add_example.Proofs.Mathematical_FiniteIndices
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedFP32Defined
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedSmallDefined

namespace VeriTile.Bench.TritonBenchG.AddExample.Finite
open VeriTile.Triton FP.Structural VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost VectorAddition.Finite
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def scalarParameters : ValueProofs.Parameters := ⟨⟨"x"⟩,⟨"y"⟩,⟨"output"⟩,1,1024⟩
def scalarOperands (a c : Cell Word) : RegionName → Nat → Cell Word :=
  fun region _ => if region=scalarParameters.x then a else c

/-- Canonical loaded words, checked on actual active loads only. -/
def InputsDefined : (r : Route) → Carrier r → Carrier r → Prop
  | .wide x y,a,c => (FP64Memory.readCell x a).isSome ∧ (FP64Memory.readCell y c).isSome
  | Route.integer x y,a,c => SharedScalarView (Route.integer x y).signature.1 a ∧ SharedScalarView (Route.integer x y).signature.2 c
  | Route.floating x y,a,c => SharedScalarView (Route.floating x y).signature.1 a ∧ SharedScalarView (Route.floating x y).signature.2 c
  | Route.mixed32 first d,a,c => SharedScalarView (Route.mixed32 first d).signature.1 a ∧ SharedScalarView (Route.mixed32 first d).signature.2 c
  | Route.mixedSmall f first d,a,c => SharedScalarView (Route.mixedSmall f first d).signature.1 a ∧ SharedScalarView (Route.mixedSmall f first d).signature.2 c

/-- No selected fallback policy may manufacture a source result for an
undefined float-to-integer conversion. The fp64 path is already partial. -/
def ConversionDefined (b : Numerics) : (r : Route) → Carrier r → Carrier r → Prop
  | .mixed32 first d,a,c => first=Bool.true →
      MixedFP32.ConversionDefined d.storage
        (MixedFP32.sumValue b.cfg first d.storage scalarParameters (scalarOperands a c) 0)
  | .mixedSmall f first d,a,c => first=Bool.true →
      MixedSmall.ConversionDefined d.storage
        (MixedSmall.conversionInput f (MixedSmall.sumValue b.cfg f first d.storage scalarParameters (scalarOperands a c) 0))
  | .wide x y,a,c => (scalarSource b (.wide x y) a c).isSome
  | _,_,_ => True

/-- Independent scalar formula: the zero fallback is unreachable under the
explicit conversion-defined check. Every floating primitive is inherited. -/
def scalarFormula (b : Numerics) : (r : Route) → Carrier r → Carrier r → Carrier r
  | .integer x y,a,c => IntegerSource.resultCell x.storage y.storage scalarParameters (scalarOperands a c) 0
  | .floating x y,a,c => TypedFloat.resultCell b.rest x.dtype y.dtype (TypedFloat.promote x y).dtype
      scalarParameters (scalarOperands a c) 0
  | .mixed32 first d,a,c => MixedFP32.resultCell b.cfg (fun _ _ => 0) (fun _ _ => 0) first d.storage
      scalarParameters (scalarOperands a c) 0
  | .mixedSmall f first d,a,c => MixedSmall.resultCell b.cfg f (fun _ _ => 0) (fun _ _ => 0) first d.storage
      scalarParameters (scalarOperands a c) 0
  | .wide x y,a,c => do
      let av ← FP64Memory.readCell x a
      let cv ← FP64Memory.readCell y c
      let value ← FP64TypedCore.result b.wide x y av cv
      pure (FP64Memory.Cell.mk x value)

theorem scalar_policy_independent (b : Numerics) (r : Route) (a c : Carrier r)
    (defined : ConversionDefined b r a c) : scalarSource b r a c=scalarFormula b r a c := by
  cases r with
  | integer x y => rfl
  | floating x y => rfl
  | wide x y => rfl
  | mixed32 first d =>
    exact MixedFP32.resultCell_policy_independent b.cfg b.undefinedSigned (fun _ _ => 0)
      b.undefinedUnsigned (fun _ _ => 0) first d.storage scalarParameters (scalarOperands a c) 0 defined
  | mixedSmall f first d =>
    exact MixedSmall.resultCell_policy_independent b.cfg f b.undefinedSigned (fun _ _ => 0)
      b.undefinedUnsigned (fun _ _ => 0) first d.storage scalarParameters (scalarOperands a c) 0 defined

inductive ScalarIssue where
  | malformedInput
  | undefinedConversion
  deriving DecidableEq, Repr

def evaluateScalar (b : Numerics) (r : Route) (a c : Carrier r) : Except ScalarIssue (Carrier r) :=
  if InputsDefined r a c then
    if ConversionDefined b r a c then .ok (scalarSource b r a c) else .error .undefinedConversion
  else .error .malformedInput

def expectedScalar (b : Numerics) (r : Route) (a c : Carrier r) : Except ScalarIssue (Carrier r) :=
  if InputsDefined r a c then
    if ConversionDefined b r a c then .ok (scalarFormula b r a c) else .error .undefinedConversion
  else .error .malformedInput

theorem scalar_correspondence (b : Numerics) (r : Route) (a c : Carrier r) :
    evaluateScalar b r a c=expectedScalar b r a c := by
  unfold evaluateScalar expectedScalar
  split
  · split
    · rename_i defined
      rw [scalar_policy_independent b r a c defined]
    · rfl
  · rfl

theorem scalar_success_defined (b : Numerics) (r : Route) (a c value : Carrier r)
    (h : evaluateScalar b r a c=.ok value) :
    InputsDefined r a c ∧ ConversionDefined b r a c ∧ value=scalarFormula b r a c := by
  rw [scalar_correspondence] at h
  unfold expectedScalar at h
  split at h
  · rename_i inputs
    split at h
    · rename_i defined
      cases h
      exact ⟨inputs,defined,rfl⟩
    · cases h
  · cases h

#specNonCircular scalarFormula avoiding [evaluateScalar,execute,runLive,runLoaded]
#specNonCircular expectedScalar avoiding [evaluateScalar,execute,runLive,runLoaded]
#axiomsClean scalar_policy_independent
#axiomsClean scalar_correspondence
#axiomsClean scalar_success_defined
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AddExample.Finite
