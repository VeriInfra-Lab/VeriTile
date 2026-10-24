import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelRealPhysical
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.RealGrid
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelInputBinding
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Direct
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelAddresses

/-! Optional checker, checked-wrapper and Tensor-pointer applications.
The complete canonical source results remain in Correct/FPEquiv. -/
namespace VeriTile.Bench.Repairs.QuantizeKvCopy.TensorBridge
noncomputable section

theorem python_checks_iff (m : Option Inputs) (w : Nat → MetadataWord) :
    checkArguments m w=.ok () ↔ Admitted m w := checkArguments_iff m w

/-- Both original output objects, exact source-order INT8/FP32 words, None and
full backing-byte frame under exactly the Python admission contract. -/
theorem quantize_kv_copy_native (actual : Operations Word Scale) (m : Inputs) (w : Nat → MetadataWord)
    (h : InputContract m w) (objects : Role → Nat) (plan : Plan (request m w))
    (before : Heap) (binding : TensorBinding m w before) :
    ∃ out, Direct.runWrapper actual m w objects plan before=.ok out ∧ out.returned=none ∧
      out.outputObject=objects .output ∧ out.scaleObject=objects .scale ∧
      out.outputMetadata=m .output ∧ out.scaleMetadata=m .scale ∧
      (∀ s a c, s<(request m w).programs → a<(request m w).groups → c<groupWidth →
        out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c)=
          codes actual plan decodeWord (directLayout m objects) before s a c) ∧
      (∀ s a (b : Fin 4), s<(request m w).programs → a<(request m w).groups →
        out.heap (scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b)=
          scales actual plan decodeWord scaleByte (directLayout m objects) before s a b) ∧
      (∀ byte, ¬Touched (request m w) (directLayout m objects) byte → out.heap byte=before byte) :=
  Direct.correct actual m w h objects plan before binding

/-- Numerical transport only: primitive abs/max/divide/scale-cast/promotion,
comparison and int8-conversion agreements at their actual operands. -/
theorem quantize_kv_copy_if_local (actual : Operations Word Scale) (undefinedCast : Word → Int)
    (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (objects : Role → Nat)
    (plan : Plan (request m w)) (before : Heap) (binding : TensorBinding m w before)
    (laws : TaskNumerics actual (softwareOps undefinedCast) (request m w) plan (directLayout m objects) before) :
    ∃ out, Direct.runWrapper actual m w objects plan before=.ok out ∧
      (∀ s a c, s<(request m w).programs → a<(request m w).groups → c<groupWidth →
        out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c)=
          codes (softwareOps undefinedCast) plan decodeWord (directLayout m objects) before s a c) ∧
      (∀ s a (b : Fin 4), s<(request m w).programs → a<(request m w).groups →
        out.heap (scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b)=
          scales (softwareOps undefinedCast) plan decodeWord scaleByte (directLayout m objects) before s a b) := by
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
theorem quantize_kv_copy_defined_int8 (actual : Operations Word Scale) (undefinedCast : Word → Int)
    (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (objects : Role → Nat)
    (plan : Plan (request m w)) (before : Heap) (binding : TensorBinding m w before)
    (laws : TaskNumerics actual (softwareOps undefinedCast) (request m w) plan (directLayout m objects) before)
    (defined : TaskCastDomain (softwareOps undefinedCast) (request m w) plan (directLayout m objects) before) :
    ∃ out, Direct.runWrapper actual m w objects plan before=.ok out ∧
      ∀ s a c, s<(request m w).programs → a<(request m w).groups → c<groupWidth →
        ∃ q z, (quotient (softwareOps undefinedCast) plan.tree
          (loadInput decodeWord (request m w) (directLayout m objects) before s a) c).toRat?=some q ∧
          VeriTile.Triton.FP.IntegerConversion.TowardZero q z ∧ VeriTile.Triton.FP.IntegerConversion.SignedRange 8 z ∧
          RowwiseQuantizationTriton.signedByte
            (out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c))=z := by
  obtain ⟨out,run,code,_⟩ := quantize_kv_copy_if_local actual undefinedCast m w h objects plan before binding laws
  refine ⟨out,run,?_⟩
  intro s a c hs ha hc
  rw [code s a c hs ha hc]
  exact reference_code_defined undefinedCast plan.tree _ c (defined s a c hs ha hc)

/-- The same source indices and caller data pointers agree with native integer
words. Ordinary Tensor storage bounds derive nonwrapping active addresses. -/
theorem quantize_kv_copy_canonical_pointers
    (t : KernelTensor.Tensors) (laws : ∀ r,KernelTensor.Laws (t r)) (before : Heap)
    (h : InputContract (KernelTensor.metadata t) (KernelTensor.words t before)) (pid g c : Nat)
    (hp : pid<(request (KernelTensor.metadata t) (KernelTensor.words t before)).programs)
    (hg : g<(request (KernelTensor.metadata t) (KernelTensor.words t before)).groups) (hc : c<8) :
    let m := KernelTensor.metadata t
    let w := KernelTensor.words t before
    (Native.pointer (base m .input) (Native.inputIndex (request m w) pid g c) 4).toNat=
      base m .input+4*Kernel.inputOffset (KernelWhole.parameters m w) pid g c ∧
    (Native.pointer (base m .output) (Native.outputIndex (request m w) ((request m w).destination pid) g c) 1).toNat=
      base m .output+Kernel.codeOffset (KernelWhole.parameters m w) ((request m w).destination pid) g c ∧
    ∀ b : Fin 4,
    (Native.pointer (base m .scale) (Native.scaleIndex (request m w) ((request m w).destination pid) g) 4).toNat+b.val=
      base m .scale+4*Kernel.scaleOffset (KernelWhole.parameters m w) ((request m w).destination pid) g+b.val :=
  KernelAddresses.pointers t laws before h pid g c hp hg hc

/-- Arbitrary independent Real source/scale cells and exact signed-byte codes.
Neither the input nor the scale target passes through a finite byte decoder. -/
theorem quantize_kv_copy_real (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (objects : Role → Nat) (plan : Plan (request m w)) (before : Mathematical.State) :
    Mathematical.Quantized (request m w) (directLayout m objects) before
      (Mathematical.run (request m w) plan (directLayout m objects) before) :=
  Mathematical.correct (request m w) (checked_domain m w h) plan (directLayout m objects) before

/-- Complete canonical DSL and whole grid over arbitrary Real input/scale
cells at actual caller Tensor addresses. The same source-final state supplies
both logical outputs, signed code bytes, Real scales and exterior frames.
This is a separate mathematical interpretation, not an FP32 accuracy claim. -/
theorem quantize_kv_copy_canonical_real
    (t : KernelTensor.Tensors) (laws : ∀ r,KernelTensor.Laws (t r)) (before : Heap)
    (h : InputContract (KernelTensor.metadata t) (KernelTensor.words t before))
    (realMemory : Nat → ℝ) (codeMemory : Nat → BitVec 8) :
    let m := KernelTensor.metadata t
    let w := KernelTensor.words t before
    let initial := KernelPublic.realInitial m w realMemory codeMemory
    ∃ final,KernelGrid.run KernelModels.realOps (KernelWhole.parameters m w)
      (List.range (request m w).programs) initial=some final ∧
      KernelWhole.Result m w
        (fun pid g c => (KernelModels.realCode (KernelPublic.realValues m w realMemory pid g) c).toInt)
        (fun pid g => KernelModels.realScale (KernelPublic.realValues m w realMemory pid g)) initial final ∧
      KernelRealPhysical.Result t w realMemory (KernelRealPhysical.scalePost m w realMemory final)
        codeMemory (KernelRealPhysical.codePost m w codeMemory final) :=
  KernelRealPhysical.correct t laws before h realMemory codeMemory


/-- Complete canonical DSL endpoint. Native scalar operations and the selected
local reduction tree are explicit compiler/ISA boundaries. No finite-value,
conversion-definedness, or output-equality assumption restricts admission.
The actual mixed-dtype caller Tensors share one backing byte heap. -/
theorem quantize_kv_copy_canonical_native
    (a : KernelModels.NativeInstructions) (tree : KernelModels.ReductionTree)
    (t : KernelTensor.Tensors) (laws : ∀ r,KernelTensor.Laws (t r)) (before : Heap)
    (h : InputContract (KernelTensor.metadata t) (KernelTensor.words t before)) :
    ∃ out,KernelPublic.nativeCall a tree t before=some out ∧
      KernelPublic.NativeResult a tree t before out ∧ KernelInputBinding.SourceInputs t before := by
  obtain ⟨out,run,result⟩ := KernelPublic.native a tree t laws before h
  exact ⟨out,run,result,KernelInputBinding.actual t laws before h⟩

theorem quantize_kv_copy_canonical_empty
    (a : KernelModels.NativeInstructions) (tree : KernelModels.ReductionTree)
    (t : KernelTensor.Tensors) (before : Heap)
    (h : InputContract (KernelTensor.metadata t) (KernelTensor.words t before))
    (empty : (request (KernelTensor.metadata t) (KernelTensor.words t before)).tokens=0) :
    ∃ out,KernelPublic.nativeCall a tree t before=some out ∧ out.tensors=t ∧ out.returned=none ∧ out.heap=before :=
  KernelPublic.empty_native a tree t before h empty


#axiomsClean python_checks_iff
#axiomsClean quantize_kv_copy_native
#axiomsClean quantize_kv_copy_if_local
#axiomsClean quantize_kv_copy_defined_int8
#axiomsClean quantize_kv_copy_canonical_pointers
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.TensorBridge
