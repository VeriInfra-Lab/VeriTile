import bench.tritonbench_g.lora_expand_gemv.Proofs.FPTypedSource

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
open QuantizeKvTransform
noncomputable section
abbrev Word := FP.Value .fp32
abbrev Half := MatmulDequantInt4.Binary16.Value
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def bfloat (x : Word) : FP.Value .bf16 := ⟨x.bits.truncate 16⟩
def embedBfloat (x : FP.Value .bf16) : Word := ⟨x.bits.zeroExtend 32⟩
def inputFormat : InputProfile → OutputFormat
  | .halfHalf | .halfBfloat => .half
  | .bfloatHalf | .bfloatBfloat => .bfloat
def weightFormat : InputProfile → OutputFormat
  | .halfHalf | .bfloatHalf => .half
  | .halfBfloat | .bfloatBfloat => .bfloat
def productFormat : InputProfile → OutputFormat
  | .bfloatBfloat => .bfloat
  | _ => .half

def decodeWord (cfg : FP.Config) : OutputFormat → Word → FP.Decoded
  | .half, x => (MatmulDequantInt4.Binary16.prepare cfg (MixedBits.half x)).decode
  | .bfloat, x => (FP.prepare cfg (bfloat x)).decode
  | .single, x => (FP.prepare cfg x).decode

def encodeWord (cfg : FP.Config) : OutputFormat → FP.Decoded → Word
  | .half, .nan .. => MixedBits.embed MatmulDequantInt4.Binary16.Value.qNaN
  | .half, .infinity s => MixedBits.embed (MatmulDequantInt4.Binary16.Value.infinity s)
  | .half, .finite s q => MixedBits.embed (MatmulDequantInt4.Binary16.round cfg (if s then -q else q) s)
  | .bfloat, .nan .. => embedBfloat (FP.Value.qNaN .bf16)
  | .bfloat, .infinity s => embedBfloat (FP.Value.infinity .bf16 s)
  | .bfloat, .finite s q => embedBfloat (FP.round .bf16 cfg (if s then -q else q) s)
  | .single, .nan .. => FP.Value.qNaN .fp32
  | .single, .infinity s => FP.Value.infinity .fp32 s
  | .single, .finite s q => FP.round .fp32 cfg (if s then -q else q) s

/-- Direct conversion through the exact decoded value. Same-format conversion
preserves bits; cross-format NaNs canonicalize explicitly. No double rounding. -/
def castWord (cfg : FP.Config) (src dst : OutputFormat) (x : Word) : Word :=
  if src = dst then x else encodeWord cfg dst (decodeWord cfg src x)

def addWord (cfg : FP.Config) : OutputFormat → Word → Word → Word
  | .half, x, y => MixedBits.embed (MatmulDequantInt4.Binary16.add cfg (MixedBits.half x) (MixedBits.half y))
  | .bfloat, x, y => embedBfloat (FP.add cfg (bfloat x) (bfloat y))
  | .single, x, y => FP.add cfg x y

def mulWord (cfg : FP.Config) : OutputFormat → Word → Word → Word
  | .half, x, y => MixedBits.embed (MatmulDequantInt4.Binary16.mul cfg (MixedBits.half x) (MixedBits.half y))
  | .bfloat, x, y => embedBfloat (FP.mul cfg (bfloat x) (bfloat y))
  | .single, x, y => FP.mul cfg x y

/-- Every input product and every explicit seed is retained in the supplied
valid addition tree. This is not a real sum or a reassociation theorem. -/
def treeValue (cfg : FP.Config) (fmt : OutputFormat) (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => FP.Value.zero .fp32
  | .add a b => addWord cfg fmt (treeValue cfg fmt xs a) (treeValue cfg fmt xs b)

def sumValues (cfg : FP.Config) (fmt : OutputFormat) (plans : ∀ n, ReductionPlan n)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Word .real shape) :
    Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => treeValue cfg fmt (fun i => xs (TileShape.insertAxisIndex shape ax out i)) (plans _).tree
  · exact fun out => treeValue cfg fmt (fun i => xs (TileShape.replaceAxisIndex shape ax out i)) (plans _).tree

def formatOf : FloatDType → Option OutputFormat
  | .fp16 => some .half | .bf16 => some .bfloat | .fp32 | .real => some .single | _ => none

def typedSum (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (d : FloatDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Word d.toTileDType shape) :
    Option (Values Word d.toTileDType (TileShape.reduceShape shape ax keep)) :=
  match formatOf d with
  | none => none
  | some fmt => some (fun out => ofFloat d
      (sumValues cfg fmt plans ax keep (fun i => toFloat d (xs i)) out))

/-- Only the used storage formats and operations are bound. Unused operations
remain in rest; the execution result will be independent of that fallback. -/
def backend (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) : Algebra Word :=
  { rest with
    literal := fun p d x => if x = 0 then FP.Value.zero .fp32 else rest.literal p d x
    binary := fun p d op x y => match formatOf d, op with
      | some fmt, .mul => mulWord cfg fmt x y
      | some fmt, .add => addWord cfg fmt x y
      | _, _ => rest.binary p d op x y
    cast := fun p src dst x => match formatOf src, formatOf dst with
      | some a, some b => castWord cfg a b x
      | _, _ => rest.cast p src dst x
    reduceSumTyped := fun _ d => typedSum cfg plans d }

@[simp] theorem castWord_same (cfg : FP.Config) (fmt : OutputFormat) (x : Word) :
    castWord cfg fmt fmt x = x := by simp [castWord]
@[simp] theorem castWord_zero (cfg : FP.Config) (src dst : OutputFormat) :
    castWord cfg src dst (FP.Value.zero .fp32) = FP.Value.zero .fp32 := by
  rcases cfg with ⟨r, fi, fr⟩
  cases src <;> cases dst <;> cases r <;> cases fi <;> cases fr <;> decide +kernel

/-- Local Triton3.8 floating sum keeps the product dtype; bf16→half promotion
occurs before a mixed half/bf16 multiplication. These checks distinguish both
low-precision multiplications and their stores from a fake fp32 product. -/
theorem format_boundary_vectors :
    mulWord .ieeeValue .half (FP.Value.ofBits .fp32 0x3c01) (FP.Value.ofBits .fp32 0x3c01) =
      FP.Value.ofBits .fp32 0x3c02 ∧
    mulWord .ieeeValue .bfloat (FP.Value.ofBits .fp32 0x3f81) (FP.Value.ofBits .fp32 0x3f81) =
      FP.Value.ofBits .fp32 0x3f82 ∧
    castWord .ieeeValue .bfloat .half (FP.Value.ofBits .fp32 0x3f80) =
      FP.Value.ofBits .fp32 0x3c00 ∧
    castWord .ieeeValue .half .single (FP.Value.ofBits .fp32 0x8000) =
      FP.Value.ofBits .fp32 0x80000000 := by
  decide +kernel

#axiomsClean castWord_same
#axiomsClean castWord_zero
#axiomsClean format_boundary_vectors
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
