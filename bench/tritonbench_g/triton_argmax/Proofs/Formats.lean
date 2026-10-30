import bench.tritonbench_g.triton_argmax.Proofs.Tree

/-! Explicit integer/bool store graph. The graph keeps each FP32 primitive and
the final truncating cast separate. Bool storage is an i8 conversion followed
by a nonzero byte load; it is not FP32 != 0. Floating formats retain their
original single store conversion. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Formats
open Source Reduction

inductive IntegerKind where
  | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | boolean
  deriving DecidableEq,Repr

def IntegerKind.bits : IntegerKind → Nat
  | .i8 | .u8 | .boolean => 8
  | .i16 | .u16 => 16
  | .i32 | .u32 => 32
  | .i64 | .u64 => 64

def IntegerKind.signed : IntegerKind → Bool
  | .i8 | .i16 | .i32 | .i64 | .boolean => Bool.true
  | _ => Bool.false

def IntegerKind.exponent (k : IntegerKind) : Nat := if k.signed then k.bits-1 else k.bits

def IntegerKind.lower (k : IntegerKind) : Int := if k.signed then -(2^(k.exponent) : Int) else 0

def IntegerKind.gap (k : IntegerKind) : Nat := 2^(k.exponent-24)

def IntegerKind.upper (k : IntegerKind) : Int := (2^(k.exponent) : Int)-k.gap

def IntegerKind.padding (k : IntegerKind) : Int := if k=.boolean then 0 else k.lower

/-- Callbacks are individual FP32 primitives and the reached float-to-integer
cast. No callback returns a reduction winner, output, or execution result. -/
structure StorePrimitives (k : IntegerKind) where
  eq32 : Word → Word → Bool
  zero32 : Word
  lower32 : Word
  upper32 : Word
  maximum32 : Word → Word → Word
  minimum32 : Word → Word → Word
  truncate : Word → BitVec k.bits

def store (k : IntegerKind) (p : StorePrimitives k) (x : Word) : BitVec k.bits :=
  let clean := if p.eq32 x x then x else p.zero32
  let lowerClamped := p.maximum32 clean p.lower32
  let upperClamped := p.minimum32 lowerClamped p.upper32
  p.truncate upperClamped

def boolLoad (byte : BitVec 8) : Bool := decide (byte.toNat ≠ 0)

def IntegerKind.dtypeName : IntegerKind → String
  | .i8 => "torch.int8"
  | .i16 => "torch.int16"
  | .i32 => "torch.int32"
  | .i64 => "torch.int64"
  | .u8 => "torch.uint8"
  | .u16 => "torch.uint16"
  | .u32 => "torch.uint32"
  | .u64 => "torch.uint64"
  | .boolean => "torch.bool"

def format (k : IntegerKind) : MidFormat :=
  ⟨k.dtypeName,k.bits/8,by cases k <;> decide⟩

theorem format_width (k : IntegerKind) : k.bits=8*(format k).bytes := by
  cases k <;> decide

def decodeStored (k : IntegerKind) (x : MidWord (format k)) : Int :=
  if k=.boolean then (if x.toNat=0 then 0 else 1)
  else if k.signed then x.toInt else Int.ofNat x.toNat

def paddingWord (k : IntegerKind) : MidWord (format k) :=
  BitVec.ofInt _ k.padding

/-- Definitional binding of the general raw graph to every integer/bool
intermediate. The only remaining callbacks are FP32 primitive instructions. -/
def integerScalars (k : IntegerKind) (p : StorePrimitives k)
    (gt32 : Word → Word → Bool) (negInf32 : Word) : Scalars (format k) where
  gt32 := gt32
  eq32 := p.eq32
  negInf32 := negInf32
  storeCast := fun x => (store k p x).cast (format_width k)
  gtMid := fun x y => orderedGt (decodeStored k x) (decodeStored k y)
  eqMid := fun x y => orderedEq (decodeStored k x) (decodeStored k y)
  negInfMid := paddingWord k

theorem padding_minimum (k : IntegerKind) (x : MidWord (format k)) :
    decodeStored k (paddingWord k) ≤ decodeStored k x := by
  cases k <;>
    simp [decodeStored,paddingWord,IntegerKind.padding,IntegerKind.lower,IntegerKind.exponent,
      IntegerKind.signed,IntegerKind.bits,format]
  all_goals first | exact BitVec.le_toInt x | split <;> omega

theorem pick_map {α β : Type} {n : Nat} (f : α → β) (gt eq : β → β → Bool)
    (tree : Tree n) (xs : Fin n → α) :
    tree.pick (fun x y => gt (f x) (f y)) (fun x y => eq (f x) (f y)) xs=
      tree.pick gt eq (fun i => f (xs i)) := by
  induction tree with
  | leaf _ => rfl
  | fork a b ha hb =>
    simp only [Tree.pick,ha,hb,leftWins]
    rfl

theorem bool_fractional_zero : boolLoad (BitVec.ofNat 8 0)=Bool.false := by decide

theorem integer_bounds (k : IntegerKind) :
    k.lower ≤ 0 ∧ 0 ≤ k.upper ∧ k.upper < (2^k.exponent : Int) := by
  cases k <;> norm_num [IntegerKind.lower,IntegerKind.upper,IntegerKind.exponent,
    IntegerKind.signed,IntegerKind.bits,IntegerKind.gap]

/-- Scalar clamp facts apply after the explicit NaN/infinity sanitization and
are independent of reduction/selection. -/
def clamp (lower upper x : Int) : Int := min (max x lower) upper

theorem clamp_bounds (lower upper x : Int) (ordered : lower ≤ upper) :
    lower ≤ clamp lower upper x ∧ clamp lower upper x ≤ upper := by
  simp only [clamp]
  exact ⟨le_min (le_max_right _ _) ordered,min_le_right _ _⟩

theorem clamp_preserves (lower upper x : Int) (lo : lower ≤ x) (hi : x ≤ upper) :
    clamp lower upper x=x := by simp [clamp,max_eq_left lo,min_eq_left hi]

/-- Neutral padding cannot win over an earlier active lane when the actual
stored intermediate values have an ordinary linear order. This covers signed,
unsigned and bool storage; no claim about NaN floating order is made here. -/
theorem padding_cannot_win {α : Type} [LinearOrder α] {n : Nat}
    (tree : Tree n) (complete : tree.Complete) (active : Nat) (nonempty : 0 < active)
    (fits : active ≤ n) (padding : α) (values : Fin n → α)
    (inactive : ∀ i : Fin n,active ≤ i.val → values i=padding)
    (neutral : ∀ i : Fin n,i.val < active → padding ≤ values i) :
    (tree.pick orderedGt orderedEq values).val < active := by
  let first : Fin n := ⟨0,by omega⟩
  let selected := tree.pick orderedGt orderedEq values
  have maximum := mathematical_maximum values tree complete
  by_contra h
  have padded : values selected=padding := inactive selected (by omega)
  have lower : padding ≤ values first := neutral first nonempty
  have upper : values first ≤ padding := padded ▸ maximum.1 first
  have equal : values first=values selected := (le_antisymm upper lower).trans padded.symm
  have tie := maximum.2 first equal
  change selected.val ≤ 0 at tie
  omega

theorem integer_padding_cannot_win (k : IntegerKind) (p : StorePrimitives k)
    (gt32 : Word → Word → Bool) (negInf32 : Word) (c : GlobalConfig)
    (complete : c.secondTree.Complete) (nonempty : 0 < c.blocks) (fits : c.blocks ≤ c.midBlock)
    (mid : Nat → MidWord (format k)) :
    (stage2Selected (integerScalars k p gt32 negInf32) c mid).val < c.blocks := by
  unfold stage2Selected
  change (c.secondTree.pick (fun x y => orderedGt (decodeStored k x) (decodeStored k y))
    (fun x y => orderedEq (decodeStored k x) (decodeStored k y)) _).val < c.blocks
  rw [pick_map]
  apply padding_cannot_win c.secondTree complete c.blocks nonempty fits (decodeStored k (paddingWord k))
  · intro i inactive
    simp [stage2Loaded,show ¬i.val < c.blocks by omega,integerScalars]
  · intro i active
    exact padding_minimum k _

#axiomsClean integer_bounds
#axiomsClean clamp_bounds
#axiomsClean clamp_preserves
#axiomsClean padding_minimum
#axiomsClean padding_cannot_win
#axiomsClean integer_padding_cannot_win
end VeriTile.Bench.Repairs.TritonArgmax.Formats
