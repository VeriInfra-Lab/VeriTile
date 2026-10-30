import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR

/-!
Execution regressions for composed natural-number bitwise expressions.
These cover the existing Nat AST only: signed i64 lowering, machine-width
casts and the original direct-Philox frontend require separate proofs.
No property of the opaque floating algebra is assumed.
-/
namespace VeriTile.Bench.Tests.TypedNatBitwise
open VeriTile.Triton

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

private theorem real_bitAnd {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOp (.bitAnd bc x y) s =
      ((evalOp x s).bind fun u => (evalOp y s).map fun v =>
        Tile.bop (fun m n : Nat => m &&& n) bc u v) := by
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy]

private theorem real_bitOr {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOp (.bitOr bc x y) s =
      ((evalOp x s).bind fun u => (evalOp y s).map fun v =>
        Tile.bop (fun m n : Nat => m ||| n) bc u v) := by
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy]

private theorem real_bitXor {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOp (.bitXor bc x y) s =
      ((evalOp x s).bind fun u => (evalOp y s).map fun v =>
        Tile.bop (fun m n : Nat => m ^^^ n) bc u v) := by
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy]

private theorem real_shiftLeft {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOp (.shiftLeft bc x y) s =
      ((evalOp x s).bind fun u => (evalOp y s).map fun v =>
        Tile.bop (fun m n : Nat => m <<< n) bc u v) := by
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy]

private theorem real_shiftRight {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOp (.shiftRight bc x y) s =
      ((evalOp x s).bind fun u => (evalOp y s).map fun v =>
        Tile.bop (fun m n : Nat => m >>> n) bc u v) := by
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy]

private theorem rounded_bitAnd (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOpR R (.bitAnd bc x y) s =
      ((evalOpR R x s).bind fun u => (evalOpR R y s).map fun v =>
        Tile.bop (fun m n : Nat => m &&& n) bc u v) := by
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy]

private theorem rounded_bitOr (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOpR R (.bitOr bc x y) s =
      ((evalOpR R x s).bind fun u => (evalOpR R y s).map fun v =>
        Tile.bop (fun m n : Nat => m ||| n) bc u v) := by
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy]

private theorem rounded_bitXor (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOpR R (.bitXor bc x y) s =
      ((evalOpR R x s).bind fun u => (evalOpR R y s).map fun v =>
        Tile.bop (fun m n : Nat => m ^^^ n) bc u v) := by
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy]

private theorem rounded_shiftLeft (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOpR R (.shiftLeft bc x y) s =
      ((evalOpR R x s).bind fun u => (evalOpR R y s).map fun v =>
        Tile.bop (fun m n : Nat => m <<< n) bc u v) := by
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy]

private theorem rounded_shiftRight (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOpR R (.shiftRight bc x y) s =
      ((evalOpR R x s).bind fun u => (evalOpR R y s).map fun v =>
        Tile.bop (fun m n : Nat => m >>> n) bc u v) := by
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy]

private theorem rounded_constNat (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (.constNat n) s = some (Tile.scalar n) := by simp only [evalOpR]

private theorem rounded_arange (R : RoundingModel) (B : Nat) (s : BlockState) :
    evalOpR R (.arange B) s = some (Tile.vec (fun i => i.val)) := by simp only [evalOpR]

private theorem rounded_add (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) (s : BlockState) :
    evalOpR R (.add .nat bc x y) s =
      ((evalOpR R x s).bind fun u => (evalOpR R y s).map fun v =>
        Tile.bop (fun m n : Nat => m + n) bc u v) := by
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy, Tile.bop, NumericDType.add]

def wordMask : Nat := 0xffffffff

def lowWord (offset : Op .nat []) : Op .nat [] :=
  .bitAnd .nil offset (.constNat wordMask)

def highWord (offset : Op .nat []) : Op .nat [] :=
  .bitAnd .nil (.shiftRight .nil offset (.constNat 32)) (.constNat wordMask)

/-- Add lane indices to the low word, mask the addition, and retain the
original high word. The low-word wrap must not carry into the high word. -/
def laneCounter (B : Nat) (offset : Op .nat []) : Op .nat [B] :=
  .bitOr .scalarR
    (.bitAnd .scalarR
      (.add .nat .scalarL (lowWord offset) (.arange B)) (.constNat wordMask))
    (.shiftLeft .nil (highWord offset) (.constNat 32))

def laneCounterValue (offset lane : Nat) : Nat :=
  (((offset &&& wordMask) + lane) &&& wordMask) |||
    (((offset >>> 32) &&& wordMask) <<< 32)

/-- An additional scalar/vector XOR exercises the opposite broadcast side.
This is an operation regression, not a claimed Philox round. -/
def keyedCounter (B : Nat) (offset key : Op .nat []) : Op .nat [B] :=
  .bitXor .scalarL (.shiftLeft .nil key (.constNat 16)) (laneCounter B offset)

private theorem lowWord_real (s : BlockState) (offset : Nat) :
    evalOp (lowWord (.constNat offset)) s = some (Tile.scalar (offset &&& wordMask)) := by
  simp only [lowWord, real_bitAnd, evalOp_constNat, Option.bind, Option.map_some,
    Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

private theorem highWord_real (s : BlockState) (offset : Nat) :
    evalOp (highWord (.constNat offset)) s =
      some (Tile.scalar ((offset >>> 32) &&& wordMask)) := by
  simp only [highWord, real_bitAnd, real_shiftRight, evalOp_constNat, Option.bind,
    Option.map_some, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

private theorem counterLow_real (s : BlockState) (B offset : Nat) :
    evalOp (.bitAnd .scalarR
      (.add .nat .scalarL (lowWord (.constNat offset)) (.arange B)) (.constNat wordMask)) s =
      some (⟨fun i => ((offset &&& wordMask) + i.1.val) &&& wordMask⟩ : Tile .nat [B]) := by
  rw [real_bitAnd, evalOp_add, lowWord_real, evalOp_arange, evalOp_constNat]
  rfl

private theorem counterHigh_real (s : BlockState) (offset : Nat) :
    evalOp (.shiftLeft .nil (highWord (.constNat offset)) (.constNat 32)) s =
      some (Tile.scalar (((offset >>> 32) &&& wordMask) <<< 32)) := by
  rw [real_shiftLeft, highWord_real, evalOp_constNat]
  rfl

private theorem lowWord_rounded (R : RoundingModel) (s : BlockState) (offset : Nat) :
    evalOpR R (lowWord (.constNat offset)) s = some (Tile.scalar (offset &&& wordMask)) := by
  simp only [lowWord, rounded_bitAnd, rounded_constNat, Option.bind, Option.map_some,
    Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

private theorem highWord_rounded (R : RoundingModel) (s : BlockState) (offset : Nat) :
    evalOpR R (highWord (.constNat offset)) s =
      some (Tile.scalar ((offset >>> 32) &&& wordMask)) := by
  simp only [highWord, rounded_bitAnd, rounded_shiftRight, rounded_constNat, Option.bind,
    Option.map_some, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

private theorem counterLow_rounded (R : RoundingModel) (s : BlockState) (B offset : Nat) :
    evalOpR R (.bitAnd .scalarR
      (.add .nat .scalarL (lowWord (.constNat offset)) (.arange B)) (.constNat wordMask)) s =
      some (⟨fun i => ((offset &&& wordMask) + i.1.val) &&& wordMask⟩ : Tile .nat [B]) := by
  rw [rounded_bitAnd, rounded_add, lowWord_rounded, rounded_arange, rounded_constNat]
  rfl

private theorem counterHigh_rounded (R : RoundingModel) (s : BlockState) (offset : Nat) :
    evalOpR R (.shiftLeft .nil (highWord (.constNat offset)) (.constNat 32)) s =
      some (Tile.scalar (((offset >>> 32) &&& wordMask) <<< 32)) := by
  rw [rounded_shiftLeft, highWord_rounded, rounded_constNat]
  rfl

theorem counter_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (B offset : Nat) :
    FP.Structural.evalOp M p (laneCounter B (.constNat offset)) s =
      some (fun i => laneCounterValue offset i.1.val) := by
  rfl

theorem counter_real (s : BlockState) (B offset : Nat) :
    evalOp (laneCounter B (.constNat offset)) s =
      some ⟨fun i => laneCounterValue offset i.1.val⟩ := by
  unfold laneCounter
  rw [real_bitOr, counterLow_real, counterHigh_real]
  rfl

theorem counter_rounded (R : RoundingModel) (s : BlockState) (B offset : Nat) :
    evalOpR R (laneCounter B (.constNat offset)) s =
      some ⟨fun i => laneCounterValue offset i.1.val⟩ := by
  unfold laneCounter
  rw [rounded_bitOr, counterLow_rounded, counterHigh_rounded]
  rfl

theorem counter_evaluators_agree {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (typed : FP.Structural.State α) (real : BlockState)
    (R : RoundingModel) (B offset : Nat) :
    FP.Structural.evalOp M p (laneCounter B (.constNat offset)) typed =
      (evalOp (laneCounter B (.constNat offset)) real).map Tile.data ∧
    FP.Structural.evalOp M p (laneCounter B (.constNat offset)) typed =
      (evalOpR R (laneCounter B (.constNat offset)) real).map Tile.data := by
  rw [counter_typed, counter_real, counter_rounded]
  exact ⟨rfl, rfl⟩

theorem keyed_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (B offset key : Nat) :
    FP.Structural.evalOp M p
      (keyedCounter B (.constNat offset) (.constNat key)) s =
      some (fun i => (key <<< 16) ^^^ laneCounterValue offset i.1.val) := by
  rfl

theorem keyed_evaluators_agree {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (typed : FP.Structural.State α) (real : BlockState)
    (R : RoundingModel) (B offset key : Nat) :
    FP.Structural.evalOp M p (keyedCounter B (.constNat offset) (.constNat key)) typed =
      (evalOp (keyedCounter B (.constNat offset) (.constNat key)) real).map Tile.data ∧
    FP.Structural.evalOp M p (keyedCounter B (.constNat offset) (.constNat key)) typed =
      (evalOpR R (keyedCounter B (.constNat offset) (.constNat key)) real).map Tile.data := by
  rw [keyed_typed]
  simp only [keyedCounter, real_bitXor, rounded_bitXor, real_shiftLeft, rounded_shiftLeft,
    evalOp_constNat, rounded_constNat, counter_real, counter_rounded,
    Option.bind, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, Option.map_some, and_self]

theorem concrete_wrap_words :
    (List.range 4).map (laneCounterValue 0x3fffffffe) =
      [0x3fffffffe, 0x3ffffffff, 0x300000000, 0x300000001] := by
  decide +kernel

theorem wrap_is_not_full_width_addition :
    laneCounterValue 0x3fffffffe 2 ≠ 0x3fffffffe + 2 := by
  decide +kernel

/-- Both tensor dimensions expand: [2,1] XOR [1,3] gives [2,3]. -/
def gridXor : Op .nat [2, 3] :=
  .bitXor (.consR (.consL .nil))
    (.ref .nat [2, 1] "rows") (.ref .nat [1, 3] "columns")

theorem grid_broadcast {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (row : Fin 2 → Nat) (col : Fin 3 → Nat)
    (hr : s.regs .nat [2, 1] "rows" = some (fun i => row i.1))
    (hc : s.regs .nat [1, 3] "columns" = some (fun i => col i.2.1)) :
    FP.Structural.evalOp M p gridXor s =
      some (fun i => row i.1 ^^^ col i.2.1) := by
  simp only [gridXor, FP.Structural.evalOp, hr, hc, bind, Option.bind]
  rfl

/-- Empty output does not erase the failing scalar offset read nested under
mask, shift, addition and word reconstruction. -/
theorem missing_offset_empty {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (h : s.regs .nat [] "offset" = none) :
    FP.Structural.evalOp M p (laneCounter 0 (.ref .nat [] "offset")) s = none := by
  simp [laneCounter, lowWord, FP.Structural.evalOp, h]

theorem missing_key_empty {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (h : s.regs .nat [] "key" = none) :
    FP.Structural.evalOp M p
      (keyedCounter 0 (.constNat 0) (.ref .nat [] "key")) s = none := by
  simp [keyedCounter, FP.Structural.evalOp, h]

/-- A successfully evaluated left XOR operand must not hide a failed right
operand, even when the resulting vector has no lanes. -/
theorem missing_right_operand_empty {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (h : s.regs .nat [] "offset" = none) :
    FP.Structural.evalOp M p
      (keyedCounter 0 (.ref .nat [] "offset") (.constNat 0)) s = none := by
  simp [keyedCounter, laneCounter, lowWord, FP.Structural.evalOp, h]

/-- A zero mask has no short-circuit semantics for missing right operands. -/
theorem zero_mask_checks_right {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (h : s.regs .nat [0] "data" = none) :
    FP.Structural.evalOp M p
      (.bitAnd .scalarL (.constNat 0) (.ref .nat [0] "data")) s = none := by
  simp [FP.Structural.evalOp, h]

theorem shift_checks_right {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (h : s.regs .nat [] "shift" = none) :
    FP.Structural.evalOp M p
      (.bitOr .scalarR (.arange 0)
        (.shiftRight .nil (.constNat 0) (.ref .nat [] "shift"))) s = none := by
  simp [FP.Structural.evalOp, h]

#print axioms counter_typed
#print axioms counter_real
#print axioms counter_rounded
#print axioms counter_evaluators_agree
#print axioms keyed_typed
#print axioms keyed_evaluators_agree
#print axioms concrete_wrap_words
#print axioms wrap_is_not_full_width_addition
#print axioms grid_broadcast
#print axioms missing_offset_empty
#print axioms missing_key_empty
#print axioms missing_right_operand_empty
#print axioms zero_mask_checks_right
#print axioms shift_checks_right
end VeriTile.Bench.Tests.TypedNatBitwise
