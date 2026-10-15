import bench.tritonbench_g.matrix_reduction.Proofs.TypedSource
import VeriTile.Triton.Float.Structural
import Mathlib.Tactic
import bench.tritonbench_g.matrix_reduction.Proofs.OrderKey
import VeriTile.Meta.StatementAudit

/-! Finite binary16/binary32 bit-pattern order and maximum-selection lemmas.
The concrete matrix backend separately implements actual32-bit intermediate
encodings. NaN payload/zero-sign choices of native unindexed maximum remain a
compiler/device refinement obligation. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax
open VeriTile.Triton FP.Structural TypedSource

def width : Storage → Nat | .fp16 => 16 | .fp32 => 32
def fractionBits : Storage → Nat | .fp16 => 10 | .fp32 => 23
def exponentBits : Storage → Nat | .fp16 => 5 | .fp32 => 8
abbrev Bits (storage : Storage) := BitVec (width storage)

def sign (storage : Storage) (x : Bits storage) : Bool :=
  decide (2 ^ (width storage - 1) ≤ x.toNat)
def magnitude (storage : Storage) (x : Bits storage) : Nat :=
  x.toNat % 2 ^ (width storage - 1)
def isNaN (storage : Storage) (x : Bits storage) : Bool :=
  decide (magnitude storage x / 2 ^ fractionBits storage = 2 ^ exponentBits storage - 1 ∧
    magnitude storage x % 2 ^ fractionBits storage ≠ 0)
/-- The monotone IEEE sign-magnitude comparison key for non-NaN encodings.
Both signed zeros have the same comparison key; infinities are included. -/
def rank : (storage : Storage) → Bits storage → Nat
  | .fp16, x => (BinaryConversion.key16 x).toNat
  | .fp32, x => (BinaryConversion.key32 x).toNat

def negativeInfinity (storage : Storage) : Bits storage :=
  BitVec.ofNat (width storage)
    (2 ^ (width storage - 1) + (2 ^ exponentBits storage - 1) * 2 ^ fractionBits storage)

def select {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (best next : Fin n) : Fin n :=
  if rank storage (xs best) < rank storage (xs next) ∨
      (rank storage (xs best) = rank storage (xs next) ∧ next.val < best.val)
    then next else best

def best {n : Nat} (storage : Storage) (xs : Fin n → Bits storage) (hn : 0 < n) : Fin n :=
  (List.finRange n).foldl (select storage xs) ⟨0, hn⟩

private theorem fold_bound {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (l : List (Fin n)) (start : Fin n) :
    rank storage (xs start) ≤ rank storage (xs (l.foldl (select storage xs) start)) ∧
    ∀ i ∈ l, rank storage (xs i) ≤ rank storage (xs (l.foldl (select storage xs) start)) := by
  induction l generalizing start with
  | nil => simp
  | cons a l ih =>
    have ha : rank storage (xs a) ≤ rank storage (xs (select storage xs start a)) := by
      unfold select
      split <;> omega
    have hs : rank storage (xs start) ≤ rank storage (xs (select storage xs start a)) := by
      unfold select
      split <;> omega
    obtain ⟨hinit, hlist⟩ := ih (select storage xs start a)
    constructor
    · exact hs.trans hinit
    · intro i hi
      rcases List.mem_cons.mp hi with rfl | hi
      · exact ha.trans hinit
      · exact hlist i hi

theorem best_maximal {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (hn : 0 < n) (i : Fin n) :
    rank storage (xs i) ≤ rank storage (xs (best storage xs hn)) :=
  (fold_bound storage xs (List.finRange n) ⟨0, hn⟩).2 i (by simp)

private def Dominates {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (a b : Fin n) : Prop :=
  rank storage (xs b) ≤ rank storage (xs a) ∧
    (rank storage (xs b) = rank storage (xs a) → a.val ≤ b.val)

private theorem dominates_trans {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    {a b c : Fin n} (hab : Dominates storage xs a b) (hbc : Dominates storage xs b c) :
    Dominates storage xs a c := by
  unfold Dominates at *
  constructor <;> omega

private theorem select_dominates {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (a b : Fin n) :
    Dominates storage xs (select storage xs a b) a ∧
    Dominates storage xs (select storage xs a b) b := by
  unfold select Dominates
  split <;> constructor <;> constructor <;> omega

private theorem fold_dominates {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (l : List (Fin n)) (start : Fin n) :
    Dominates storage xs (l.foldl (select storage xs) start) start ∧
      ∀ i ∈ l, Dominates storage xs (l.foldl (select storage xs) start) i := by
  induction l generalizing start with
  | nil => simp [Dominates]
  | cons a l ih =>
    obtain ⟨hinit, hlist⟩ := ih (select storage xs start a)
    obtain ⟨hs, ha⟩ := select_dominates storage xs start a
    constructor
    · exact dominates_trans storage xs hinit hs
    · intro i hi
      rcases List.mem_cons.mp hi with rfl | hi
      · exact dominates_trans storage xs hinit ha
      · exact hlist i hi

theorem best_leftmost {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (hn : 0 < n) (i : Fin n)
    (he : rank storage (xs i) = rank storage (xs (best storage xs hn))) :
    (best storage xs hn).val ≤ i.val :=
  ((fold_dominates storage xs (List.finRange n) ⟨0, hn⟩).2 i (by simp)).2 he

/-- An independent output property: a source bit pattern with no smaller
machine comparison key than any input. It does not mention execution/reduction. -/
def IsMaximum {n : Nat} (storage : Storage) (xs : Fin n → Bits storage)
    (out : Bits storage) : Prop :=
  (∃ i, out = xs i) ∧ ∀ i, rank storage (xs i) ≤ rank storage out

theorem best_spec {n : Nat} (storage : Storage) (xs : Fin n → Bits storage) (hn : 0 < n) :
    IsMaximum storage xs (xs (best storage xs hn)) :=
  ⟨⟨best storage xs hn, rfl⟩, best_maximal storage xs hn⟩

/-- All bits are executable. The maximum-value interpretation is promised on
non-NaN inputs; NaNs have a deterministic software ordering only. Signed-zero
outputs retain a selected input's exact bits. -/
def rowMaximum {n : Nat} (storage : Storage) (xs : Fin n → Bits storage) : Bits storage :=
  if hn : 0 < n then xs (best storage xs hn) else negativeInfinity storage

def reduce (storage : Storage) {shape : TileShape} (ax : Fin shape.length) :
    (keep : Bool) → Values (Bits storage) .real shape →
    Values (Bits storage) .real (TileShape.reduceShape shape ax keep)
  | Bool.false, xs => fun out => rowMaximum storage
      (fun k => xs (TileShape.insertAxisIndex shape ax out k))
  | Bool.true, xs => fun out => rowMaximum storage
      (fun k => xs (TileShape.replaceAxisIndex shape ax out k))

#axiomsClean best_maximal
#axiomsClean best_spec
#axiomsClean best_leftmost
end VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax
