import bench.tritonbench_g.triton_attention.Proofs.FPPreprocess
import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
open AttentionKernelAligned.FPValue (Plans)
abbrev Word := FP.Value .fp32
abbrev Half := MatmulDequantInt4.Binary16.Value
inductive StorageFormat where | half | bfloat | single deriving DecidableEq, Repr
def StorageFormat.dtype : StorageFormat → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .fp32
def formatOf : FloatDType → Option StorageFormat
  | .fp16 => some .half | .bf16 => some .bfloat | .fp32 | .real => some .single | _ => none
def decodeWord : StorageFormat → Word → FP.Decoded
  | .half,x => MatmulDequantInt4.Binary16.Value.decode ⟨x.bits.truncate 16⟩
  | .bfloat,x => (⟨x.bits.truncate 16⟩ : FP.Value .bf16).decode
  | .single,x => x.decode
def encodeWord : StorageFormat → FP.Decoded → Word
  | .half,.nan .. => ⟨MatmulDequantInt4.Binary16.Value.qNaN.bits.zeroExtend 32⟩
  | .half,.infinity s => ⟨(MatmulDequantInt4.Binary16.Value.infinity s).bits.zeroExtend 32⟩
  | .half,.finite s q => ⟨(MatmulDequantInt4.Binary16.round .ieeeValue (if s then -q else q) s).bits.zeroExtend 32⟩
  | .bfloat,.nan .. => ⟨(FP.Value.qNaN .bf16).bits.zeroExtend 32⟩
  | .bfloat,.infinity s => ⟨(FP.Value.infinity .bf16 s).bits.zeroExtend 32⟩
  | .bfloat,.finite s q => ⟨(FP.round .bf16 .ieeeValue (if s then -q else q) s).bits.zeroExtend 32⟩
  | .single,.nan .. => FP.Value.qNaN .fp32
  | .single,.infinity s => FP.Value.infinity .fp32 s
  | .single,.finite s q => FP.round .fp32 .ieeeValue (if s then -q else q) s
def castWord (src dst : StorageFormat) (x : Word) : Word :=
  if src=dst then x else encodeWord dst (decodeWord src x)

/-- These are individual operation bindings, not identities or execution/output hypotheses. -/
structure External where
  hostLiteral : Option ComputeDType → FloatDType → ℝ → Word
  exp : Option ComputeDType → Word → Word
  dot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values Word left.toTileDType (batch++[m,k]) → Values Word right.toTileDType (batch++[k,n]) →
    Values Word out.toTileDType (batch++[m,n])
  otherCast : Option ComputeDType → FloatDType → FloatDType → Word → Word

def convert (external : External) (p : Option ComputeDType) (src dst : FloatDType) (x : Word) : Word :=
  match formatOf src,formatOf dst with
  | some a,some b => castWord a b x
  | _,_ => external.otherCast p src dst x
noncomputable def software (external : External) (plans : Plans) : Primitives Word :=
  { literal := fun p d x => if x=0 then FP.Value.zero .fp32 else
      if x=1 then FP.Value.ofBits .fp32 0x3f800000 else external.hostLiteral p d x
    negInf := FP.Value.infinity .fp32 Bool.true
    binary := fun _ d op x y => match d,op with
      | .real,.add | .fp32,.add => FP.add .ieeeValue x y
      | .real,.sub | .fp32,.sub => FP.sub .ieeeValue x y
      | .real,.mul | .fp32,.mul => FP.mul .ieeeValue x y
      | .real,.div | .fp32,.div => FP.div .ieeeValue x y
      | .real,.max | .fp32,.max => AttentionKernelAligned.FPValue.maxnum x y
      | _,_ => FP.Value.qNaN .fp32
    unary := fun p op x => if op=.exp then external.exp p x else FP.Value.qNaN .fp32
    cast := convert external
    max := fun _ {_} ax keep v => AttentionKernelAligned.FPValue.reduceValues plans.maximumTree
      AttentionKernelAligned.FPValue.maxnum (FP.Value.infinity .fp32 Bool.true) ax keep v
    sum := fun _ {_} ax keep v => AttentionKernelAligned.FPValue.reduceValues plans.sumTree
      (FP.add .ieeeValue) (FP.Value.zero .fp32) ax keep v
    dot := external.dot }

/-- Within the three declared preprocess storage profiles, no external primitive
can influence its initial-memory reference. Zero denominators/NaNs remain in domain. -/
theorem preprocess_external_independent [Inhabited Word] (a b : External) (plans : Plans)
    (p : PreParameters) (initial : State Word) (fmt : StorageFormat) (h : p.dtype=fmt.dtype) :
    preWrite (software a plans) p initial initial=preWrite (software b plans) p initial initial := by
  cases fmt <;>
    delta preWrite preGradientCell preDeltaCell preDelta preProducts preQuotient preInput preDenominator <;>
    simp [software,convert,h,formatOf,StorageFormat.dtype]

theorem conversion_boundary_vectors :
    castWord .half .single ⟨1⟩=⟨0x33800000⟩ ∧
    castWord .single .half ⟨0x3f800001⟩=⟨0x3c00⟩ ∧
    castWord .bfloat .single ⟨0x3f80⟩=⟨0x3f800000⟩ ∧
    castWord .single .bfloat ⟨0x3f818000⟩=⟨0x3f82⟩ ∧
    castWord .single .half ⟨0x80000000⟩=⟨0x8000⟩ := by decide +kernel

#axiomsClean preprocess_external_independent
#axiomsClean conversion_boundary_vectors
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
