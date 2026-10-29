import VeriTile.Triton.Float.ScalarOpsLaws
import Mathlib.Tactic

/-! Exact FP32 exponential special values in an explicit NVIDIA specification
model. This is a special-value specification, not an implementation of the
finite nonzero approximation algorithm and not a hardware conformance proof.

PTX `ex2.approx.f32` specifies -Inf -> +0, both zero signs -> +1,
+Inf -> +Inf, and NaN -> NaN. CUDA 13.0 `expf`, whose libdevice entry is
`__nv_expf`, specifies the same table. The table is executable below; neither
target rewrite is a field supplied by the caller. The residual finite algorithm
and NaN result word remain explicit, and the two intrinsic identities have
different types. NaN payload equality is not asserted.

Triton v3.7.1 FP32 `tl.exp` lowers to multiplication by the FP32 word
0x3fb8aa3b followed by `llvm.nvvm.ex2.approx.f32`. That multiplication is kept
as `FP.mul .ieeeValue`, including its signed-zero and exceptional behavior.
Only the original FP32 profile is covered here; BF16 and FP64 do not inherit
these results. See `documents/FloatingPointSpecialValues.md` for sources.
-/
namespace VeriTile.Triton.FP.ExponentialExact

abbrev Word := Value .fp32

inductive Primitive where
  | ptxEx2Approx | libdeviceExpf
  deriving DecidableEq

/-- The parts not specified by the special-value table. Distinct indices keep
`ex2.approx.f32` and `__nv_expf` separate. The NaN result may have any payload
or sign; the specification only requires that it is a NaN. No accuracy claim
about `finiteNonzero` is made by this deliberately partial model. -/
structure Residual (primitive : Primitive) where
  finiteNonzero : Word → Word
  nan : Word → { y : Word // y.isNaN = true }

def positiveOne : Word := Value.ofBits .fp32 0x3f800000

/-- The documented full special-value dispatch, shared only as a specification
table. Its callers retain distinct primitive identities and residual code. -/
def specialValueModel {p : Primitive} (rest : Residual p) (x : Word) : Word :=
  if x.isNaN then (rest.nan x).val
  else if x.isInf then
    if x.sign then Value.zero .fp32 else Value.infinity .fp32
  else if x.isZero then positiveOne
  else rest.finiteNonzero x

def ex2Approx (rest : Residual .ptxEx2Approx) (x : Word) : Word :=
  specialValueModel rest x

def libdeviceExp (rest : Residual .libdeviceExpf) (x : Word) : Word :=
  specialValueModel rest x

def log2e : Word := Value.ofBits .fp32 0x3fb8aa3b

def tlExp (rest : Residual .ptxEx2Approx) (x : Word) : Word :=
  ex2Approx rest (FP.mul .ieeeValue x log2e)

/-- Includes both signed zeros and every subnormal. -/
def finite (x : Word) : Prop := x.exponent < 255

theorem finite_iff_toRat (x : Word) : finite x ↔ x.toRat?.isSome = true := by
  have bound : x.exponent < 256 := Nat.mod_lt _ (by decide)
  by_cases h : x.exponent = 255
  · by_cases hf : x.fraction = 0 <;> simp [finite, Value.toRat?, Value.decode, h, hf]
  · have hx : x.exponent < 255 := by omega
    simp [finite, Value.toRat?, Value.decode, h, hx]

private theorem neg_exponent (x : Word) : x.neg.exponent = x.exponent := by
  have bound := x.bits.isLt
  simp only [Value.exponent, Value.neg, Value.ofBits, Value.raw, Value.sign,
    Format.fractionBits, Format.width, BitVec.toNat_ofNat, bne_iff_ne]
  norm_num at *
  split_ifs <;> omega

private theorem prepare_finite (cfg : Config) (x : Word) (hx : finite x) :
    finite (prepare cfg x) := by
  unfold prepare
  split
  · change (Value.zero .fp32 x.sign).exponent < 255
    cases hs : x.sign <;> decide +kernel
  · exact hx

private theorem finite_decode (x : Word) (hx : finite x) :
    ∃ s q, x.decode = .finite s q := by
  have h : x.exponent ≠ 255 := Nat.ne_of_lt hx
  simp [Value.decode, h]

/-- The subtraction part is a theorem of the existing scalar software model,
for every rounding/flush configuration and every finite FP32 input word. -/
theorem neg_inf_sub (cfg : Config) (x : Word) (hx : finite x) :
    FP.sub cfg (Value.infinity .fp32 true) x = Value.infinity .fp32 true := by
  have hn : finite x.neg := by simpa only [finite, neg_exponent] using hx
  obtain ⟨s, q, hq⟩ := finite_decode _ (prepare_finite cfg x.neg hn)
  have hi : (prepare cfg (Value.infinity .fp32 true)).decode = .infinity true := by
    have hn : (Value.infinity .fp32 true).isSubnormal = false := by decide +kernel
    simp only [prepare, hn, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
    decide +kernel
  simp only [FP.sub, FP.add, hi, hq]

theorem special_zero {p : Primitive} (rest : Residual p) (s : Bool) :
    specialValueModel rest (Value.zero .fp32 s) = positiveOne := by
  cases s <;> rfl

theorem special_neg_inf {p : Primitive} (rest : Residual p) :
    specialValueModel rest (Value.infinity .fp32 true) = Value.zero .fp32 := rfl

theorem special_pos_inf {p : Primitive} (rest : Residual p) :
    specialValueModel rest (Value.infinity .fp32) = Value.infinity .fp32 := rfl

theorem special_nan {p : Primitive} (rest : Residual p) (x : Word)
    (hx : x.isNaN = true) : (specialValueModel rest x).isNaN = true := by
  simpa only [specialValueModel, hx, ↓reduceIte] using (rest.nan x).property

theorem tl_exp_zero (rest : Residual .ptxEx2Approx) (s : Bool) :
    tlExp rest (Value.zero .fp32 s) = positiveOne := by
  have hm : FP.mul .ieeeValue (Value.zero .fp32 s) log2e = Value.zero .fp32 s := by
    cases s <;> decide +kernel
  simpa only [tlExp, hm, ex2Approx] using special_zero rest s

theorem tl_exp_neg_inf (rest : Residual .ptxEx2Approx) :
    tlExp rest (Value.infinity .fp32 true) = Value.zero .fp32 := by
  have hm : FP.mul .ieeeValue (Value.infinity .fp32 true) log2e =
      Value.infinity .fp32 true := by decide +kernel
  simpa only [tlExp, hm, ex2Approx] using special_neg_inf rest

theorem libdevice_exp_zero (rest : Residual .libdeviceExpf) (s : Bool) :
    libdeviceExp rest (Value.zero .fp32 s) = positiveOne := special_zero rest s

theorem libdevice_exp_neg_inf (rest : Residual .libdeviceExpf) :
    libdeviceExp rest (Value.infinity .fp32 true) = Value.zero .fp32 := special_neg_inf rest

theorem tl_exp_neg_inf_sub (rest : Residual .ptxEx2Approx) (cfg : Config)
    (x : Word) (hx : finite x) :
    tlExp rest (FP.sub cfg (Value.infinity .fp32 true) x) = Value.zero .fp32 := by
  rw [neg_inf_sub cfg x hx, tl_exp_neg_inf]

theorem libdevice_exp_neg_inf_sub (rest : Residual .libdeviceExpf) (cfg : Config)
    (x : Word) (hx : finite x) :
    libdeviceExp rest (FP.sub cfg (Value.infinity .fp32 true) x) = Value.zero .fp32 := by
  rw [neg_inf_sub cfg x hx, libdevice_exp_neg_inf]

#axiomsClean finite_iff_toRat
#axiomsClean neg_inf_sub
#axiomsClean special_zero
#axiomsClean special_neg_inf
#axiomsClean special_pos_inf
#axiomsClean special_nan
#axiomsClean tl_exp_zero
#axiomsClean tl_exp_neg_inf
#axiomsClean libdevice_exp_zero
#axiomsClean libdevice_exp_neg_inf
#axiomsClean tl_exp_neg_inf_sub
#axiomsClean libdevice_exp_neg_inf_sub

end VeriTile.Triton.FP.ExponentialExact
