import bench.repairs.tritonbench_g.fused_activation.Proofs.Graph
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.FusedActivation.Words
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := Graph.Word
def finite (word : Word) : Prop := ((⟨word⟩ : FP.Value .fp32).toRat?).isSome

/-- Exact raw-word reference ports for the original selected FP32 approximate
exp2 and full division primitives. These are deliberately opaque, independent
of `native` and NOT implemented or correctly-rounded exp2/division functions.
This optional model-conformance target is not a numerical experiment obligation. -/
structure OriginalPrimitiveReference where
  expFP32 : Word → Word
  exp2ApproxFP32 : Word → Word
  divisionFullFP32 : Word → Word → Word

def softwareFMA (input x : Word) : Word :=
  (FP.fma FP.Config.ieeeValue (⟨input⟩ : FP.Value .fp32) ⟨Graph.half⟩ ⟨x⟩).bits

def softwareAdd (x y : Word) : Word :=
  (FP.add FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩).bits

/-- The original sigmoid adds one first; the backend emits one second. This
is a concrete software-word theorem, including exceptional input patterns. -/
theorem softwareAdd_one_comm (x : Word) : softwareAdd x Graph.one=softwareAdd Graph.one x := by
  exact congrArg FP.Value.bits (FP.ScalarOpsLaws.add_one_comm_fp32 ⟨x⟩)

def softwareSub (x y : Word) : Word :=
  (FP.sub FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩).bits

def softwareMul (x y : Word) : Word :=
  (FP.mul FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩).bits

/-- Optional primitive-model conformance, outside the required kernel rewrite.
These hypotheses select a software interpretation; they are not deferred
numerical transformation experiments. The production graph remains opaque
at these primitive boundaries, and ADD-COMMUTE requires none of these laws. -/
structure LocalExact (native : Graph.Operations) (reference : OriginalPrimitiveReference) : Prop where
  fma : ∀ input x, finite input → finite x → native.fmaRN32 input Graph.half x=softwareFMA input x
  add : ∀ x y, finite x → finite y → native.add32 x y=softwareAdd x y
  negative : ∀ z, finite z → native.sub32 Graph.zero z=softwareSub Graph.zero z
  scale : ∀ z, finite z → native.mul32 z Graph.log2e=softwareMul z Graph.log2e
  exp2 : ∀ z, native.ex2Approx32 z=reference.exp2ApproxFP32 z
  division : ∀ numerator denominator, native.divFull32 numerator denominator=reference.divisionFullFP32 numerator denominator

def loweredReferenceWord (reference : OriginalPrimitiveReference) (x bias input : Word) : Word :=
  let z := softwareAdd (softwareFMA input x) bias
  let negative := softwareSub Graph.zero z
  let scaled := softwareMul negative Graph.log2e
  let exponent := reference.exp2ApproxFP32 scaled
  reference.divisionFullFP32 Graph.one (softwareAdd exponent Graph.one)

/-- Domain of the actual local basic-operation operands. Finite leaves alone
are insufficient: FMA/add can overflow, and exp2 can overflow before add-one.
Outside this numerical domain the unconditional native-word theorem remains. -/
def NumericDomain (native : Graph.Operations) (x bias input : Word) : Prop :=
  let fused := native.fmaRN32 input Graph.half x
  let z := native.add32 fused bias
  let negative := native.sub32 Graph.zero z
  let exponent := native.ex2Approx32 (native.mul32 negative Graph.log2e)
  finite input ∧ finite x ∧ finite bias ∧ finite fused ∧ finite z ∧ finite negative ∧ finite exponent

theorem local_exact (native : Graph.Operations) (reference : OriginalPrimitiveReference)
    (laws : LocalExact native reference) (x bias input : Word) (domain : NumericDomain native x bias input) :
    Graph.nativeReference native Bool.true x bias input=loweredReferenceWord reference x bias input := by
  obtain ⟨hi,hx,hb,hf,hz,hn,he⟩ := domain
  have fma := laws.fma input x hi hx
  have add := laws.add _ bias hf hb
  have neg := laws.negative _ hz
  have scale := laws.scale _ hn
  have oneFinite : finite Graph.one := by
    change ((⟨Graph.one⟩ : FP.Value .fp32).toRat?).isSome=Bool.true
    decide
  have plus := laws.add _ Graph.one he oneFinite
  simp only [Graph.nativeReference,↓reduceIte,Graph.sigmoid,loweredReferenceWord]
  rw [laws.division,plus,laws.exp2,scale,neg,add,fma]

/-- Source-level FP reference expressed independently as 1/(1+exp(-z)),
with the original opaque tl.exp FP32 port. It is not a Real exponential or a
correctly-rounded software implementation. -/
def referenceWord (reference : OriginalPrimitiveReference) (x bias input : Word) : Word :=
  let z := softwareAdd (softwareFMA input x) bias
  reference.divisionFullFP32 Graph.one
    (softwareAdd Graph.one (reference.expFP32 (softwareSub Graph.zero z)))

/-- Optional interpretation of the tl.exp primitive by a scaled approximate
exp2 instruction. This is a compiler/primitive-model boundary, not a source
transformation performed by this task and not a numerical experiment TODO. -/
def LocalExpLowering (reference : OriginalPrimitiveReference) : Prop :=
  ∀ z, finite z → reference.exp2ApproxFP32 (softwareMul z Graph.log2e)=reference.expFP32 z

theorem source_reference_exact (native : Graph.Operations) (reference : OriginalPrimitiveReference)
    (laws : LocalExact native reference) (lowering : LocalExpLowering reference)
    (x bias input : Word) (domain : NumericDomain native x bias input) :
    Graph.nativeReference native Bool.true x bias input=referenceWord reference x bias input := by
  have lower := local_exact native reference laws x bias input domain
  obtain ⟨hi,hx,hb,hf,hz,hn,he⟩ := domain
  have fma := laws.fma input x hi hx
  have add := laws.add _ bias hf hb
  have neg := laws.negative _ hz
  rw [neg,add,fma] at hn
  rw [lower]
  simp only [loweredReferenceWord,referenceWord]
  rw [lowering _ hn,softwareAdd_one_comm]

#specNonCircular referenceWord avoiding [candidateKernel,FP.Structural.exec,runGrid,Graph.nativeReference]
#specNonCircular LocalExact avoiding [candidateKernel,FP.Structural.exec,runGrid]
#specNonCircular LocalExpLowering avoiding [candidateKernel,FP.Structural.exec,runGrid]
#axiomsClean softwareAdd_one_comm
#axiomsClean source_reference_exact
#axiomsClean local_exact
end
end VeriTile.Bench.Repairs.FusedActivation.Words
