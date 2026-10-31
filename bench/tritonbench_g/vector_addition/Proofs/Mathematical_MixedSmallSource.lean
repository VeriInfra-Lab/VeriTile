import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedFP32Source
import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_Binary16
import bench.tritonbench_g.matrix_reduction.Proofs.BinaryConversion

/-! Concrete software materializations of all 36 integer/bool ↔ fp16/bf16
profiles. fp16 rounds exact integers directly to binary16 and adds in binary16.
bf16 is extended to binary32 and added there. Integer outputs convert from
binary32, with an explicit binary16 extension where required. No native byte,
compiler, poison, trap, sanitizer or physical-address refinement is claimed. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
open VeriTile.Triton FP.Structural FP.IntegerConversion
open ValueProofs (Parameters offset)
open IntegerSource (Storage Canonical)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
abbrev Half := MatmulDequantInt4.Binary16.Value
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

inductive FloatStorage where
  | fp16 | bf16
  deriving DecidableEq, Repr

def FloatStorage.dtype : FloatStorage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16

def FloatStorage.compute : FloatStorage → FloatDType
  | .fp16 => .fp16 | .bf16 => .fp32

/-- The Word carrier stores half/bfloat payloads in its low 16 bits. Its
binary32 tag is a carrier choice, not an extra numerical conversion. -/
def half (v : Word) : Half := ⟨v.bits.truncate 16⟩
def embedHalf (v : Half) : Word := ⟨v.bits.zeroExtend 32⟩
def bfloat (v : Word) : FP.Value .bf16 := ⟨v.bits.truncate 16⟩
def embedBFloat (v : FP.Value .bf16) : Word := ⟨v.bits.zeroExtend 32⟩
def widenHalf (v : Word) : Word := ⟨MatrixReduction.BinaryConversion.widen (half v).bits⟩
/-- Exact field embedding, retaining sign, signed zero and NaN payload bits. -/
def widenBFloat (v : Word) : Word := ⟨(bfloat v).bits.zeroExtend 32 <<< 16⟩
def narrowBFloat (cfg : FP.Config) (v : Word) : Word := embedBFloat (FP.cast .bf16 cfg v)

def roundTarget (cfg : FP.Config) : FloatStorage → Rat → Word
  | .fp16, q => embedHalf (MatmulDequantInt4.Binary16.round cfg q)
  | .bf16, q => FP.round .fp32 cfg q

def addTarget (cfg : FP.Config) : FloatStorage → Word → Word → Word
  | .fp16, x, y => embedHalf (MatmulDequantInt4.Binary16.add cfg (half x) (half y))
  | .bf16, x, y => FP.add cfg x y

def extendInput : FloatStorage → Word → Word
  | .fp16, v => v | .bf16, v => widenBFloat v

def conversionInput : FloatStorage → Word → Word
  | .fp16, v => widenHalf v | .bf16, v => v

def storedFloat (cfg : FP.Config) : FloatStorage → Word → Cell Word
  | .fp16, v => .mk .fp16 v | .bf16, v => .mk .bf16 (narrowBFloat cfg v)

def readFloat : FloatStorage → Cell Word → Word
  | .fp16, c => c.read .fp16 | .bf16, c => c.read .bf16

/-- The profile selects direct integer rounding. The surrounding explicit
`.real → compute dtype` AST cast is an identity carrier-tag bridge. There is
no fp32 rounding stage on the fp16 path and no native `.real` provenance claim. -/
def backend (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) : Algebra Word :=
  fp32TotalBackend
    { rest with
      fromInt := fun p n => if p = none then roundTarget cfg f (n : Rat) else rest.fromInt p n
      fromNat := fun p n => if p = none then roundTarget cfg f (n : Rat) else rest.fromNat p n
      binary := fun p d op x y =>
        if p = none ∧ d = .fp16 ∧ op = .add then addTarget cfg .fp16 x y
        else if p = none ∧ d = .fp32 ∧ op = .add then FP.add cfg x y
        else rest.binary p d op x y
      cast := fun p src dst v =>
        if p = none ∧ src = .real ∧ dst = f.compute then v
        else if p = none ∧ src = .fp16 ∧ dst = .fp32 then widenHalf v
        else if p = none ∧ src = .bf16 ∧ dst = .fp32 then widenBFloat v
        else if p = none ∧ src = .fp32 ∧ dst = .bf16 then narrowBFloat cfg v
        else rest.cast p src dst v }
    us uu

theorem backend_add (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (a b : Word) :
    (backend rest cfg f us uu).binary none f.compute .add a b = addTarget cfg f a b := by
  cases f <;> rfl

theorem backend_add_fp16 (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (a b : Word) :
    (backend rest cfg .fp16 us uu).binary none .fp16 .add a b = addTarget cfg .fp16 a b := rfl

theorem backend_add_bf16 (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (a b : Word) :
    (backend rest cfg .bf16 us uu).binary none .fp32 .add a b = addTarget cfg .bf16 a b := rfl

theorem backend_signed (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (width : Nat) :
    (backend rest cfg f us uu).floatToInt none .fp32 width =
      some (fun v => some (totalSigned (us width) width v)) := rfl

theorem backend_unsigned (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (width : Nat) :
    (backend rest cfg f us uu).floatToNat none .fp32 width =
      some (fun v => some (totalUnsigned (uu width) width v)) := rfl

theorem backend_extend_half (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (v : Word) :
    (backend rest cfg f us uu).cast none .fp16 .fp32 v = widenHalf v := by cases f <;> rfl

theorem backend_extend_bfloat (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (v : Word) :
    (backend rest cfg f us uu).cast none .bf16 .fp32 v = widenBFloat v := by cases f <;> rfl

theorem backend_narrow_bfloat (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (v : Word) :
    (backend rest cfg f us uu).cast none .fp32 .bf16 v = narrowBFloat cfg v := by cases f <;> rfl

def FloatStorage.numeric : (f : FloatStorage) → NumericDType f.compute.toTileDType
  | .fp16 => .fp16 | .bf16 => .fp32

def integerToFloat (f : FloatStorage) (d : Storage) (e : Op .int shape) : Op f.compute.toTileDType shape :=
  .castFloat .real f.compute
    (if d.compute.signed then .intToReal (IntegerSource.castOp d.compute e)
     else .natToReal (.intToUnsigned d.compute.width e))

def integerRounded (cfg : FP.Config) (f : FloatStorage) (d : Storage) (v : Int) : Word :=
  if d.compute.signed then roundTarget cfg f ((IntegerSource.castValue d.compute v : Int) : Rat)
  else roundTarget cfg f (((BitVec.ofInt d.compute.width v).toNat : Nat) : Rat)

theorem eval_integerToFloat (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (d : Storage) (e : Op .int shape) (s : State Word) :
    FP.Structural.evalOp (backend rest cfg f us uu) none (integerToFloat f d e) s = (do
      let v ← FP.Structural.evalOp (backend rest cfg f us uu) none e s
      pure (fun i => ofFloat f.compute (integerRounded cfg f d (v i)))) := by
  cases f
  case fp16 =>
    unfold integerToFloat
    by_cases hs : d.compute.signed = Bool.true
    · simp only [hs, ↓reduceIte, FP.Structural.evalOp, MixedFP32.eval_castOp_at]
      cases he : FP.Structural.evalOp (backend rest cfg .fp16 us uu) none e s <;>
        simp [he, backend, fp32TotalBackend, integerRounded, hs, bind, pure,
          FloatStorage.compute, ofFloat, toFloat]
    · simp only [hs, ↓reduceIte, FP.Structural.evalOp, MixedFP32.eval_natToReal_at,
        FP.Structural.eval_intToUnsigned]
      cases he : FP.Structural.evalOp (backend rest cfg .fp16 us uu) none e s <;>
        simp only [backend, fp32TotalBackend, FloatStorage.compute] at he <;>
        simp [he, backend, fp32TotalBackend, integerRounded, hs, bind, pure,
          FloatStorage.compute, FP.Structural.evalOp, MixedFP32.eval_natToReal_at,
          FP.Structural.eval_intToUnsigned, bop, integralMod, IntegerExpr.same_left,
          IntegerExpr.same_right, unsigned_cast_eq_bitvec, -BitVec.toNat_ofInt,
          -BitVec.toInt_ofInt, ofFloat, toFloat]
  case bf16 =>
    unfold integerToFloat
    by_cases hs : d.compute.signed = Bool.true
    · simp only [hs, ↓reduceIte, FP.Structural.evalOp, MixedFP32.eval_castOp_at]
      cases he : FP.Structural.evalOp (backend rest cfg .bf16 us uu) none e s <;>
        simp [he, backend, fp32TotalBackend, integerRounded, hs, bind, pure,
          FloatStorage.compute, ofFloat, toFloat]
    · simp only [hs, ↓reduceIte, FP.Structural.evalOp, MixedFP32.eval_natToReal_at,
        FP.Structural.eval_intToUnsigned]
      cases he : FP.Structural.evalOp (backend rest cfg .bf16 us uu) none e s <;>
        simp only [backend, fp32TotalBackend, FloatStorage.compute] at he <;>
        simp [he, backend, fp32TotalBackend, integerRounded, hs, bind, pure,
          FloatStorage.compute, FP.Structural.evalOp, MixedFP32.eval_natToReal_at,
          FP.Structural.eval_intToUnsigned, bop, integralMod, IntegerExpr.same_left,
          IntegerExpr.same_right, unsigned_cast_eq_bitvec, -BitVec.toNat_ofInt,
          -BitVec.toInt_ofInt, ofFloat, toFloat]

theorem eval_integerToFloat_fp16 (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (d : Storage) (e : Op .int shape) (s : State Word) :
    FP.Structural.evalOp (backend rest cfg .fp16 us uu) none (show Op .fp16 shape from integerToFloat .fp16 d e) s = (do
      let v ← FP.Structural.evalOp (backend rest cfg .fp16 us uu) none e s
      pure (fun i => integerRounded cfg .fp16 d (v i))) := by
  exact eval_integerToFloat rest cfg .fp16 us uu d e s

theorem eval_integerToFloat_bf16 (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (d : Storage) (e : Op .int shape) (s : State Word) :
    FP.Structural.evalOp (backend rest cfg .bf16 us uu) none (show Op .fp32 shape from integerToFloat .bf16 d e) s = (do
      let v ← FP.Structural.evalOp (backend rest cfg .bf16 us uu) none e s
      pure (fun i => integerRounded cfg .bf16 d (v i))) := by
  exact eval_integerToFloat rest cfg .bf16 us uu d e s

def floatOperand (f : FloatStorage) (name : RegName) : Op f.compute.toTileDType shape :=
  match f with
  | .fp16 => .ref .fp16 shape name
  | .bf16 => .castFloat .bf16 .fp32 (.ref .bf16 shape name)

def conversionOperand (f : FloatStorage) : Op .fp32 shape :=
  match f with
  | .fp16 => .castFloat .fp16 .fp32 (.ref .fp16 shape "output")
  | .bf16 => .ref .fp32 shape "output"

abbrev convertedCell := MixedFP32.convertedCell

def sumValue (cfg : FP.Config) (f : FloatStorage) (integerFirst : Bool) (d : Storage)
    (p : Parameters) (mem : RegionName → Nat → Cell Word) (o : Nat) : Word :=
  if integerFirst then addTarget cfg f
    (integerRounded cfg f d (IntegerSource.loaded d (mem p.x o))) (extendInput f (readFloat f (mem p.y o)))
  else addTarget cfg f (extendInput f (readFloat f (mem p.x o)))
    (integerRounded cfg f d (IntegerSource.loaded d (mem p.y o)))

def resultCell (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters)
    (mem : RegionName → Nat → Cell Word) (o : Nat) : Cell Word :=
  let sum := sumValue cfg f integerFirst d p mem o
  if integerFirst then convertedCell us uu d (conversionInput f sum) else storedFloat cfg f sum

#axiomsClean backend_add
#axiomsClean backend_add_fp16
#axiomsClean backend_add_bf16
#axiomsClean eval_integerToFloat_fp16
#axiomsClean eval_integerToFloat_bf16
#axiomsClean backend_signed
#axiomsClean backend_unsigned
#axiomsClean backend_extend_half
#axiomsClean backend_extend_bfloat
#axiomsClean backend_narrow_bfloat
#axiomsClean eval_integerToFloat
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
