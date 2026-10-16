import bench.tritonbench_g.vector_addition_custom.Proofs.NativeArithmetic

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native
open VeriTile.Bench.TritonBenchG.VectorAddition
open VeriTile.Triton FP.Structural DTypeScope TypedHost VectorAddition.Finite
noncomputable section
open Classical
set_option maxHeartbeats 6000000
local instance : Inhabited TypedHost.Word := ⟨FP.Value.zero .fp32⟩

/-- One-cell parameters match the reused scalar formula; block extent is not consulted.
The actual kernel tile extent is16 in NativeArithmetic. -/
def scalarParameters : ValueProofs.Parameters := ⟨⟨"x"⟩,⟨"y"⟩,⟨"output"⟩,1,1024⟩
def scalarOperands (a c : Cell TypedHost.Word) : RegionName → Nat → Cell TypedHost.Word :=
  fun region _ => if region=scalarParameters.x then a else c

/-- Raw finite input encodings, including noncanonical raw boolean bytes.
These are trusted typed storage primitives' input relation, not wrapper checks. -/
def scalarInput (r : Route) (left : Bool) (value : Carrier r) : Prop :=
  match r with
  | .wide x y =>
      if left then ∃ v, value=some (FP64Memory.Cell.mk x v)
      else ∃ v, value=some (FP64Memory.Cell.mk y v)
  | .integer _ _ | .floating _ _ | .mixed32 _ _ | .mixedSmall _ _ _ =>
      SharedScalarView (if left then r.signature.1 else r.signature.2) value

/-- Undefined float-to-integer conversions are observed before a value is
materialized; they never use a numerical policy as an arbitrary fallback. -/
def scalarDefined (b : Numerics) : (r : Route) → Carrier r → Carrier r → Prop
  | .mixed32 first d,a,c => first=Bool.true → MixedFP32.ConversionDefined d.storage
      (MixedFP32.sumValue b.cfg first d.storage scalarParameters (scalarOperands a c) 0)
  | .mixedSmall f first d,a,c => first=Bool.true → MixedSmall.ConversionDefined d.storage
      (MixedSmall.conversionInput f (MixedSmall.sumValue b.cfg f first d.storage scalarParameters (scalarOperands a c) 0))
  | .wide x y,a,c => (VectorAddition.Finite.scalarSource b (.wide x y) a c).isSome
  | _,_,_ => True

/-- Independent scalar formula for the actual loaded words. Existing exact
finite addition and cast functions supply every route; output dtype is x. -/
def scalarResult (b : Numerics) : (r : Route) → Carrier r → Carrier r → Carrier r
  | .integer x y,a,c => IntegerSource.resultCell x.storage y.storage scalarParameters (scalarOperands a c) 0
  | .floating x y,a,c => TypedFloat.resultCell b.rest x.dtype y.dtype (TypedFloat.promote x y).dtype
      scalarParameters (scalarOperands a c) 0
  | .mixed32 first d,a,c => MixedFP32.resultCell b.cfg (fun _ _ => 0) (fun _ _ => 0)
      first d.storage scalarParameters (scalarOperands a c) 0
  | .mixedSmall f first d,a,c => MixedSmall.resultCell b.cfg f (fun _ _ => 0) (fun _ _ => 0)
      first d.storage scalarParameters (scalarOperands a c) 0
  | .wide x y,a,c => do
      let av ← FP64Memory.readCell x a
      let cv ← FP64Memory.readCell y c
      let v ← FP64TypedCore.result b.wide x y av cv
      pure (FP64Memory.Cell.mk x v)

theorem scalar_defined_correspondence (b : Numerics) (r : Route) (a c : Carrier r)
    (defined : scalarDefined b r a c) : VectorAddition.Finite.scalarSource b r a c=scalarResult b r a c := by
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

/-- The old mathematical prefix/value result remains a conditional refinement
of these concrete loaded-word results, rather than a bound on the master API. -/
theorem scalar_logical_refinement {σ : Type} (b : Numerics) (r : Route)
    (p : ValueProofs.Parameters) (s : World σ) (i : Nat)
    (defined : scalarDefined b r (logicalRead r s p.x i) (logicalRead r s p.y i)) :
    scalarResult b r (logicalRead r s p.x i) (logicalRead r s p.y i)=sourceCell b r p s i := by
  rw [←scalar_defined_correspondence b r _ _ defined,sourceCell_scalar]

#specNonCircular scalarResult avoiding [TypedHost.execute,VectorAddition.Finite.runLive,VectorAddition.Finite.runLoaded]
#axiomsClean scalar_defined_correspondence
#axiomsClean scalar_logical_refinement
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native
