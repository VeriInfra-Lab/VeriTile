import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_TypedGrid

/-! Finite source-word decoding for the nine integer/bool ReLU profiles.
The comparison/conversion routes follow the frozen sm90 frontend evidence:
small signed words sign-extend to i32 for comparison; small unsigned words
zero-extend to i32 and compare signed; u32/u64 compare unsigned; bool loads
an i8 word, compares !=0, and converts its resulting unsigned i1 to fp32.
This is a logical word refinement, not a byte-address, allocator, compiler,
or native floating-conversion theorem. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.IntegerWords
open VeriTile.Triton FP.Structural GatedGrid TypedValue
noncomputable section

inductive Width where
  | w8 | w16 | w32 | w64
  deriving DecidableEq, Repr

def Width.bits : Width → Nat
  | .w8 => 8 | .w16 => 16 | .w32 => 32 | .w64 => 64

inductive Encoding where
  | signed (width : Width) | unsigned (width : Width) | boolean
  deriving DecidableEq, Repr

def Encoding.bits : Encoding → Nat
  | .signed w | .unsigned w => w.bits
  | .boolean => 8

abbrev Word (e : Encoding) := BitVec e.bits

def Encoding.profile : Encoding → Profile
  | .signed .w8 => .i8 | .signed .w16 => .i16
  | .signed .w32 => .i32 | .signed .w64 => .i64
  | .unsigned .w8 => .u8 | .unsigned .w16 => .u16
  | .unsigned .w32 => .u32 | .unsigned .w64 => .u64
  | .boolean => .boolean

def Encoding.kind : Encoding → Kind
  | .signed _ => .signed | .unsigned _ => .unsigned | .boolean => .boolean

@[simp] theorem profile_kind (e : Encoding) : e.profile.kind = e.kind := by
  cases e with
  | signed w => cases w <;> rfl
  | unsigned w => cases w <;> rfl
  | boolean => rfl

/-- Normalize all 256 possible bool storage bytes, not just canonical 0/1. -/
def boolDecoded (v : BitVec 8) : Bool := decide (v ≠ 0)

def boolBits (v : BitVec 8) : BitVec 1 :=
  if boolDecoded v then 1 else 0

def decodeCell {α : Type} (e : Encoding) (v : Word e) : Cell α :=
  match e with
  | .signed _ => .mk .int v.toInt
  | .unsigned _ => .mk .nat v.toNat
  | .boolean => .mk .bool (boolDecoded v)

theorem decoded_signed_bounds (w : Width) (v : BitVec w.bits) :
    -(2 : Int) ^ (w.bits - 1) ≤ v.toInt ∧ v.toInt < (2 : Int) ^ (w.bits - 1) :=
  ⟨BitVec.le_toInt v, BitVec.toInt_lt⟩

theorem decoded_unsigned_bound (w : Width) (v : BitVec w.bits) :
    v.toNat < 2 ^ w.bits := v.isLt

/-- Signed/unsigned decoding preserves the complete source word. Bool has
the separate, deliberately many-to-one normalization theorem below. -/
theorem signed_decode_roundtrip {α : Type} [Inhabited α]
    (w : Width) (v : BitVec w.bits) :
    BitVec.ofInt w.bits ((decodeCell (α := α) (.signed w) v).read .int) = v := by
  change BitVec.ofInt w.bits v.toInt = v
  exact BitVec.ofInt_toInt

theorem unsigned_decode_roundtrip {α : Type} [Inhabited α]
    (w : Width) (v : BitVec w.bits) :
    BitVec.ofNat w.bits ((decodeCell (α := α) (.unsigned w) v).read .nat) = v := by
  change BitVec.ofNat w.bits v.toNat = v
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq]

/-- Identity at widths32/64 is expressed by signExtend at equal width. -/
def signedComparison (w : Width) (v : BitVec w.bits) : Bool :=
  (0 : BitVec (max 32 w.bits)).sle (v.signExtend (max 32 w.bits))

def unsignedComparison (w : Width) (v : BitVec w.bits) : Bool :=
  if w.bits < 32 then (0 : BitVec 32).sle (v.setWidth 32)
  else (0 : BitVec w.bits).ule v

def booleanComparison (v : BitVec 8) : Bool :=
  (0 : BitVec 32).sle ((boolBits v).setWidth 32)

theorem signed_extension_value (w : Width) (v : BitVec w.bits) :
    (v.signExtend (max 32 w.bits)).toInt = v.toInt :=
  BitVec.toInt_signExtend_of_le (Nat.le_max_right _ _)

@[simp] theorem signed_comparison (w : Width) (v : BitVec w.bits) :
    signedComparison w v = decide (0 ≤ v.toInt) := by
  simp [signedComparison, BitVec.sle_eq_decide, signed_extension_value]

theorem zero_extended_nonnegative {w : Nat} (v : BitVec w) (h : w < 32) :
    (0 : BitVec 32).sle (v.setWidth 32) = Bool.true := by
  have hv : (v.setWidth 32).toInt = (v.toNat : Int) := by
    simpa only [BitVec.setWidth'_eq] using
      (BitVec.toInt_setWidth'_of_lt (x := v) h)
  simp [BitVec.sle_eq_decide, hv]

@[simp] theorem unsigned_comparison (w : Width) (v : BitVec w.bits) :
    unsignedComparison w v = Bool.true := by
  unfold unsignedComparison
  split
  · exact zero_extended_nonnegative v ‹_›
  · simp [BitVec.ule_eq_decide]

@[simp] theorem boolean_comparison (v : BitVec 8) : booleanComparison v = Bool.true :=
  zero_extended_nonnegative (boolBits v) (by decide)

@[simp] theorem bool_bits_value (v : BitVec 8) :
    (boolBits v).toNat = if boolDecoded v then 1 else 0 := by
  unfold boolBits
  split <;> rfl

theorem bool_normalization_zero (v : BitVec 8) :
    boolDecoded v = Bool.false ↔ v = 0 := by simp [boolDecoded]

theorem bool_normalization_nonzero (v : BitVec 8) :
    boolDecoded v = Bool.true ↔ v ≠ 0 := by simp [boolDecoded]

/-- Observed integer comparison/select/conversion route over finite words.
Floating conversions and the fp32 zero literal remain explicit callbacks. -/
def wordValue {α : Type} (M : Algebra α) (e : Encoding) (v : Word e) : α :=
  match e with
  | .signed w =>
      if signedComparison w v then M.fromInt none v.toInt else M.literal none .fp32 0
  | .unsigned w =>
      if unsignedComparison w v then M.fromNat none v.toNat else M.literal none .fp32 0
  | .boolean =>
      if booleanComparison v then M.fromNat none (boolBits v).toNat else M.literal none .fp32 0

theorem scalar_refines {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (e : Encoding) (v : Word e) :
    resultValue M cmp16 cmp32 e.profile.kind (decodeCell e v) = wordValue M e v := by
  cases e <;> simp [profile_kind, Encoding.kind, resultValue, decodeCell,
    Cell.read, wordValue]
  all_goals rfl

#specNonCircular wordValue avoiding [TypedValue.source, materialized,
  runPrefix, runGrid, FP.Structural.exec]
#axiomsClean signed_extension_value
#axiomsClean decoded_signed_bounds
#axiomsClean decoded_unsigned_bound
#axiomsClean signed_decode_roundtrip
#axiomsClean unsigned_decode_roundtrip
#axiomsClean signed_comparison
#axiomsClean zero_extended_nonnegative
#axiomsClean unsigned_comparison
#axiomsClean boolean_comparison
#axiomsClean bool_bits_value
#axiomsClean bool_normalization_zero
#axiomsClean bool_normalization_nonzero
#axiomsClean scalar_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.IntegerWords
