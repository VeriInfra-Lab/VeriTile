import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Direct
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquiv
noncomputable section

/-- Both original output objects, exact source-order INT8/FP16 words, None and
full backing-byte frame under exactly the Python admission contract. -/
specification quantize_copy_kv_native (actual : Operations Word Half) (m : Inputs) (w : Nat → MetadataWord)
    (h : InputContract m w) (objects : Role → Nat) (plan : Plan (request m w))
    (before : Heap) (binding : TensorBinding m w before) :
    ∃ out, Direct.runWrapper actual m w objects plan before=.ok out ∧ out.returned=none ∧
      out.outputObject=objects .output ∧ out.scaleObject=objects .scale ∧
      out.outputMetadata=m .output ∧ out.scaleMetadata=m .scale ∧
      (∀ s a c, s<(request m w).tokens → a<(request m w).heads → c<(request m w).columns →
        out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c)=
          codes actual plan decodeWord (directLayout m objects) before s a c) ∧
      (∀ s a (b : Fin 2), s<(request m w).tokens → a<(request m w).heads →
        out.heap (scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b)=
          scales actual plan decodeWord halfByte (directLayout m objects) before s a b) ∧
      (∀ byte, ¬Touched (request m w) (directLayout m objects) byte → out.heap byte=before byte) :=
  Direct.correct actual m w h objects plan before binding

/-- Numerical transport only: primitive abs/max/divide/half-cast/promotion,
comparison and int8-conversion agreements at their actual operands. -/
theorem quantize_copy_kv_if_local (actual : Operations Word Half) (undefinedCast : Word → Int)
    (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (objects : Role → Nat)
    (plan : Plan (request m w)) (before : Heap) (binding : TensorBinding m w before)
    (laws : TaskNumerics actual (softwareOps undefinedCast) (request m w) plan (directLayout m objects) before) :
    ∃ out, Direct.runWrapper actual m w objects plan before=.ok out ∧
      (∀ s a c, s<(request m w).tokens → a<(request m w).heads → c<(request m w).columns →
        out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c)=
          codes (softwareOps undefinedCast) plan decodeWord (directLayout m objects) before s a c) ∧
      (∀ s a (b : Fin 2), s<(request m w).tokens → a<(request m w).heads →
        out.heap (scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b)=
          scales (softwareOps undefinedCast) plan decodeWord halfByte (directLayout m objects) before s a b) := by
  obtain ⟨out,run,_,_,_,_,_,q,s,_⟩ := Direct.correct actual m w h objects plan before binding
  refine ⟨out,run,?_,?_⟩
  · intro t a c ht ha hc
    exact (q t a c ht ha hc).trans ((row_numerics actual (softwareOps undefinedCast) (request m w) plan _
      (laws t a ht ha)).1 c hc)
  · intro t a b ht ha
    rw [s t a b ht ha]
    unfold scales
    rw [(row_numerics actual (softwareOps undefinedCast) (request m w) plan _ (laws t a ht ha)).2]

/-- On the explicit defined-conversion trace, every stored byte denotes the
signed, toward-zero conversion of the source-order FP32 quotient. This is a
numerical condition on the conclusion, never an extra wrapper guard. -/
theorem quantize_copy_kv_defined_int8 (actual : Operations Word Half) (undefinedCast : Word → Int)
    (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (objects : Role → Nat)
    (plan : Plan (request m w)) (before : Heap) (binding : TensorBinding m w before)
    (laws : TaskNumerics actual (softwareOps undefinedCast) (request m w) plan (directLayout m objects) before)
    (defined : TaskCastDomain (softwareOps undefinedCast) (request m w) plan (directLayout m objects) before) :
    ∃ out, Direct.runWrapper actual m w objects plan before=.ok out ∧
      ∀ s a c, s<(request m w).tokens → a<(request m w).heads → c<(request m w).columns →
        ∃ q z, (quotient (softwareOps undefinedCast) plan.tree
          (loadInput decodeWord (request m w) (directLayout m objects) before s a) c).toRat?=some q ∧
          VeriTile.Triton.FP.IntegerConversion.TowardZero q z ∧ VeriTile.Triton.FP.IntegerConversion.SignedRange 8 z ∧
          RowwiseQuantizationTriton.signedByte
            (out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c))=z := by
  obtain ⟨out,run,code,_⟩ := quantize_copy_kv_if_local actual undefinedCast m w h objects plan before binding laws
  refine ⟨out,run,?_⟩
  intro s a c hs ha hc
  rw [code s a c hs ha hc]
  exact reference_code_defined undefinedCast plan.tree _ c (defined s a c hs ha hc)

#axiomsClean quantize_copy_kv_defined_int8
#axiomsClean quantize_copy_kv_native
#axiomsClean quantize_copy_kv_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquiv
