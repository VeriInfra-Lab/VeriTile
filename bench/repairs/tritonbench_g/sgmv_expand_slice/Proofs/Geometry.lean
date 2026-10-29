import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.SgmvExpandSlice

/-- Independent dimensions; the supplied output may have extra or fewer rows,
provided every active segment fits both input and output. -/
structure Shape where
  (tokens outputRows outputColumns adapters columns inner slice batches : Nat)

structure Segment where
  (start length : Nat)
  adapter : Option Nat
  deriving DecidableEq

def Segment.active (s : Segment) : Prop := s.length > 0 ∧ s.adapter.isSome
instance (s : Segment) : Decidable s.active := inferInstanceAs (Decidable (_ ∧ _))

def ActiveCell (p : Shape) (s : Segment) (r c : Nat) : Prop :=
  s.active ∧ s.start ≤ r ∧ r < s.start + s.length ∧ p.slice ≤ c ∧ c < p.slice + p.columns
instance (p : Shape) (s : Segment) (r c : Nat) : Decidable (ActiveCell p s r c) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

structure SegmentFits (p : Shape) (s : Segment) : Prop where
  input : s.active → s.start + s.length ≤ p.tokens
  output : s.active → s.start + s.length ≤ p.outputRows
  adapter : ∀ a, s.adapter = some a → s.length > 0 → a < p.adapters

/-- A normal allocated positive-width storage. Real interpretations use logical
values independently and are never encoded into this finite byte carrier. -/
structure Allocation where
  (base elements bytesPerElement : Nat)
  positiveWidth : 0 < bytesPerElement
  signedExtent : elements * bytesPerElement < 2^63
  addressExtent : base + elements * bytesPerElement ≤ 2^64

theorem linear_lt {r c rows cols : Nat} (hr : r < rows) (hc : c < cols) :
    r * cols + c < rows * cols := by nlinarith

theorem linear_injective {r c r' c' cols : Nat} (hc : c < cols) (hc' : c' < cols)
    (h : r * cols + c = r' * cols + c') : r = r' ∧ c = c' := by
  have hd := congrArg (fun n => n / cols) h
  have hm := congrArg (fun n => n % cols) h
  have hp : 0 < cols := by omega
  simpa [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.add_mod, Nat.div_eq_of_lt hc, Nat.div_eq_of_lt hc',
    Nat.mod_eq_of_lt hc, Nat.mod_eq_of_lt hc', Nat.ne_of_gt hp] using And.intro hd hm

def safeRow (s : Segment) (localRow : Nat) :=
  if localRow < s.length then s.start + localRow else 0

def safeColumn (p : Shape) (localColumn : Nat) :=
  if localColumn < p.columns then p.slice + localColumn else p.slice

def safeInner (p : Shape) (k : Nat) := if k < p.inner then k else 0

theorem safeRow_lt (p : Shape) (s : Segment) (f : SegmentFits p s)
    (active : s.active) (r : Nat) : safeRow s r < p.outputRows := by
  have ho := f.output active
  have hl := active.1
  unfold safeRow
  split_ifs <;> omega

theorem safeColumn_lt (p : Shape) (capacity : p.slice + p.columns ≤ p.outputColumns)
    (hn : 0 < p.columns) (c : Nat) : safeColumn p c < p.outputColumns := by
  unfold safeColumn
  split_ifs <;> omega

theorem safeInner_lt (p : Shape) (hk : 0 < p.inner) (k : Nat) :
    safeInner p k < p.inner := by unfold safeInner; split_ifs <;> omega

theorem clamped_output_offset (p : Shape) (s : Segment) (f : SegmentFits p s)
    (active : s.active) (capacity : p.slice + p.columns ≤ p.outputColumns)
    (hn : 0 < p.columns) (r c : Nat) :
    safeRow s r * p.outputColumns + safeColumn p c < p.outputRows * p.outputColumns :=
  linear_lt (safeRow_lt p s f active r) (safeColumn_lt p capacity hn c)

theorem allocation_element_i64 (a : Allocation) {i : Nat} (hi : i < a.elements) :
    i < 2^63 := by
  have h := a.signedExtent
  have hw := a.positiveWidth
  nlinarith

theorem allocation_byte_address (a : Allocation) {i byte : Nat}
    (hi : i < a.elements) (hb : byte < a.bytesPerElement) :
    a.base + i * a.bytesPerElement + byte < 2^64 := by
  have h := linear_lt hi hb
  have hx := a.addressExtent
  omega

theorem low_or_single_element_bound (a : Allocation) (hw : 2 ≤ a.bytesPerElement) :
    a.elements < 2^62 := by
  have h := a.signedExtent
  norm_num at h ⊢
  nlinarith

theorem padded_coordinate_i64 {n x : Nat} (hn : n < 2^62) (hx : x < n + 32) :
    x < 2^63 := by norm_num at hn hx ⊢; omega

/-- These identities expose ordinary one-dimensional tile coverage. Native grid
acceptance and actual PID delivery remain the central runtime boundary. -/
def tiles (n : Nat) := (n + 31) / 32

theorem tiles_positive {n : Nat} (h : 0 < n) : 0 < tiles n := by
  unfold tiles; omega

theorem tile_coordinate {n x : Nat} (h : x < n) :
    x / 32 < tiles n ∧ x % 32 < 32 ∧ (x / 32) * 32 + x % 32 = x := by
  unfold tiles
  have hm := Nat.mod_lt x (by decide : 0 < 32)
  have hd := Nat.div_add_mod x 32
  omega

theorem one_dimensional_coordinate {n m r c : Nat} (hr : r < m) (hc : c < n) :
    let pid := r / 32 * tiles n + c / 32
    pid / tiles n = r / 32 ∧ pid % tiles n = c / 32 ∧ pid < tiles m * tiles n := by
  dsimp
  have hn := tiles_positive (by omega : 0 < n)
  have ht := tile_coordinate hc
  have hr' := tile_coordinate hr
  constructor
  · simpa [Nat.div_eq_of_lt ht.1, Nat.add_comm] using Nat.add_mul_div_right (c / 32) (r / 32) hn
  constructor
  · simp [Nat.add_mod, Nat.mod_eq_of_lt ht.1]
  · exact linear_lt hr'.1 ht.1

/-- Ordinary tensor-storage validity is per logical coordinate. It admits the
arbitrary ignored strides of singleton dimensions and does not impose a new
canonical-stride restriction on Torch's contiguous tensors. -/
structure MatrixView where
  allocation : Allocation
  (rows columns offset rowStride columnStride : Nat)
  coordinateBound : ∀ r c, r < rows → c < columns →
    offset + r*rowStride + c*columnStride < allocation.elements

structure WeightView where
  allocation : Allocation
  (adapters columns inner offset adapterStride columnStride innerStride : Nat)
  coordinateBound : ∀ a c k, a < adapters → c < columns → k < inner →
    offset + a*adapterStride + k*innerStride + c*columnStride < allocation.elements

theorem matrix_address_i64 (v : MatrixView) (r c : Nat) (hr : r < v.rows)
    (hc : c < v.columns) : v.offset+r*v.rowStride+c*v.columnStride < 2^63 :=
  allocation_element_i64 v.allocation (v.coordinateBound r c hr hc)

theorem matrix_intermediates_i64 (v : MatrixView) (r c : Nat) (hr : r < v.rows)
    (hc : c < v.columns) : r*v.rowStride < 2^63 ∧ c*v.columnStride < 2^63 ∧
      v.offset+r*v.rowStride < 2^63 := by
  have h := matrix_address_i64 v r c hr hc
  omega

theorem weight_address_i64 (v : WeightView) (a c k : Nat) (ha : a < v.adapters)
    (hc : c < v.columns) (hk : k < v.inner) :
    v.offset+a*v.adapterStride+k*v.innerStride+c*v.columnStride < 2^63 :=
  allocation_element_i64 v.allocation (v.coordinateBound a c k ha hc hk)

theorem gathered_row_in_bounds (p : Shape) (s : Segment) (f : SegmentFits p s)
    (active : s.active) (r : Nat) : s.start + r%s.length < p.tokens := by
  have hm := Nat.mod_lt r active.1
  have hf := f.input active
  omega

theorem gathered_column_in_bounds (p : Shape) (hn : 0 < p.columns) (c : Nat) :
    c%p.columns < p.columns := Nat.mod_lt c hn

end VeriTile.Bench.Repairs.SgmvExpandSlice
