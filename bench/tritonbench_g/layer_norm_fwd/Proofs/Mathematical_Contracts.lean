import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Formula
import VeriTile.Triton.Semantics.State

/-!
Complete mathematical memory contract for one launched row. It includes both
dropout masks, the optional residual output and Y1, Mean and Rstd, and every
unchanged tagged memory cell. This is an independent specification: no kernel
execution or readback is used in its definitions. Connecting the complete
source and candidate execution to this contract remains a proof obligation.
-/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Triton
noncomputable section

structure Regions where
  x : RegionName
  y : RegionName
  weight : RegionName
  bias : RegionName
  residual : RegionName
  x1 : RegionName
  weight1 : RegionName
  bias1 : RegionName
  y1 : RegionName
  residualOut : RegionName
  rowscale : RegionName
  seeds : RegionName
  dropoutMask : RegionName
  mean : RegionName
  rstd : RegionName

structure Strides where
  x : Nat
  y : Nat
  residual : Nat
  residualOut : Nat
  x1 : Nat
  y1 : Nat

def snapshot (r : Regions) (stride : Strides) (s : BlockState) : Inputs where
  x := fun row col => s.readMem r.x (row * stride.x + col)
  x1 := fun row col => s.readMem r.x1 (row * stride.x1 + col)
  residual := fun row col => s.readMem r.residual (row * stride.residual + col)
  weight := s.readMem r.weight
  bias := s.readMem r.bias
  weight1 := s.readMem r.weight1
  bias1 := s.readMem r.bias1
  rowscale := s.readMem r.rowscale
  seeds := s.readMemValue .int r.seeds

namespace Contract

def inRow (base N offset : Nat) : Prop := base ≤ offset ∧ offset < base + N

instance (base N offset : Nat) : Decidable (inRow base N offset) :=
  inferInstanceAs (Decidable (base ≤ offset ∧ offset < base + N))

/-- Last-store-first choice follows the actual source write order, so the
specification is explicit even before output-disjointness is applied. Values
are all calculated from the initial input snapshot. -/
def expectedCell (f : Flags) (r : Regions) (stride : Strides)
    (a : Inputs) (M N : Nat) (eps p : ℝ) (row : Nat)
    (region : RegionName) (offset : Nat) : Option MemCell :=
  if f.w1 && decide (region = r.y1 ∧ inRow (row * stride.y1) N offset) then
    some (MemCell.real (Formula.y1 f a M N eps p row (offset - row * stride.y1)))
  else if region = r.y ∧ inRow (row * stride.y) N offset then
    some (MemCell.real (Formula.y f a M N eps p row (offset - row * stride.y)))
  else if region = r.rstd ∧ offset = row then
    some (MemCell.real (Formula.rstd f a M N eps p row))
  else if !f.rms && decide (region = r.mean ∧ offset = row) then
    some (MemCell.real (Formula.mean f a M N p row))
  else if f.residualOut && decide (region = r.residualOut ∧
      inRow (row * stride.residualOut) N offset) then
    some (MemCell.real (Formula.combined f a M p row (offset - row * stride.residualOut)))
  else if f.dropout && f.dropoutMask && f.x1 &&
      decide (region = r.dropoutMask ∧ inRow ((M + row) * N) N offset) then
    some (MemCell.of .bool (Formula.keep (a.seeds (M + row)) p (offset - (M + row) * N)))
  else if f.dropout && f.dropoutMask &&
      decide (region = r.dropoutMask ∧ inRow (row * N) N offset) then
    some (MemCell.of .bool (Formula.keep (a.seeds row) p (offset - row * N)))
  else none

def written (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (region : RegionName) (offset : Nat) : Prop :=
  (f.w1 = true ∧ region = r.y1 ∧ inRow (row * stride.y1) N offset) ∨
  (region = r.y ∧ inRow (row * stride.y) N offset) ∨
  (region = r.rstd ∧ offset = row) ∨
  (f.rms = false ∧ region = r.mean ∧ offset = row) ∨
  (f.residualOut = true ∧ region = r.residualOut ∧
    inRow (row * stride.residualOut) N offset) ∨
  (f.dropout = true ∧ f.dropoutMask = true ∧ f.x1 = true ∧
    region = r.dropoutMask ∧ inRow ((M + row) * N) N offset) ∨
  (f.dropout = true ∧ f.dropoutMask = true ∧
    region = r.dropoutMask ∧ inRow (row * N) N offset)

/-- Full exact tagged memory postcondition. Inactive buffers and out-of-row
cells retain the complete MemCell, rather than just its real-valued readback. -/
def Post (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState) : Prop :=
  ∀ region offset,
    final.mem region offset =
      (expectedCell f r stride (snapshot r stride initial) M N eps p initial.pid
        region offset).getD (initial.mem region offset)

theorem expectedCell_none_iff (f : Flags) (r : Regions) (stride : Strides)
    (a : Inputs) (M N : Nat) (eps p : ℝ) (row : Nat)
    (region : RegionName) (offset : Nat) :
    expectedCell f r stride a M N eps p row region offset = none ↔
      ¬ written f r stride M N row region offset := by
  unfold expectedCell
  split_ifs <;> simp_all [written, Bool.and_eq_true]

theorem post_frame (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (initial final : BlockState)
    (h : Post f r stride M N eps p initial final)
    (region : RegionName) (offset : Nat)
    (hw : ¬ written f r stride M N initial.pid region offset) :
    final.mem region offset = initial.mem region offset := by
  rw [h]
  rw [(expectedCell_none_iff _ _ _ _ _ _ _ _ _ _ _).mpr hw]
  rfl

theorem inRow_of_col {base N col : Nat} (h : col < N) : inRow base N (base + col) := by
  unfold inRow
  omega

theorem inRow_iff_col {base N offset : Nat} :
    inRow base N offset ↔ ∃ col, col < N ∧ offset = base + col := by
  unfold inRow
  constructor
  · rintro ⟨hlo, hhi⟩
    exact ⟨offset - base, by omega, by omega⟩
  · rintro ⟨col, hcol, rfl⟩
    omega

theorem rows_disjoint {row other stride N a b : Nat}
    (hstride : N ≤ stride) (ha : a < N) (hb : b < N)
    (heq : row * stride + a = other * stride + b) : row = other := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
  · have hmul := Nat.mul_le_mul_right stride hlt
    have : (row + 1) * stride ≤ other * stride := hmul
    simp only [Nat.add_mul, Nat.one_mul] at this
    omega
  · have hmul := Nat.mul_le_mul_right stride hgt
    have : (other + 1) * stride ≤ row * stride := hmul
    simp only [Nat.add_mul, Nat.one_mul] at this
    omega

end Contract
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd
