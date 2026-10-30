import bench.tritonbench_g.triton_attention.Sources.MathStreaming

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001

/-- The admitted whole invocation: canonical contiguous FP32 storage. The
FP16 compatibility evidence is separate. A block is exactly 128 rows. -/
structure Shape where
  batches : Nat
  heads : Nat
  blocks : Nat
  dim : Nat
  batches_pos : 0 < batches
  heads_pos : 0 < heads
  blocks_pos : 0 < blocks
  dim_supported : dim = 16 ∨ dim = 32 ∨ dim = 64 ∨ dim = 128
  element_bound : batches * heads * (blocks * 128) * dim ≤ 2147483647
  grid_y_bound : batches * heads ≤ 65535

def Shape.rows (c : Shape) := c.blocks * 128
def Shape.groups (c : Shape) := c.batches * c.heads
def Shape.elements (c : Shape) := c.groups * c.rows * c.dim

theorem Shape.rows_pos (c : Shape) : 0 < c.rows := by
  have h := c.blocks_pos
  simp only [Shape.rows]; omega

theorem Shape.dim_pos (c : Shape) : 0 < c.dim := by
  rcases c.dim_supported with h | h | h | h <;> omega

abbrev Coord (c : Shape) := Fin c.groups × Fin c.rows × Fin c.dim

def offset (c : Shape) (p : Coord c) : Nat :=
  (p.1.val * c.rows + p.2.1.val) * c.dim + p.2.2.val

def rowOffset (c : Shape) (h : Fin c.groups) (i : Fin c.rows) : Nat :=
  h.val * c.rows + i.val

theorem rowOffset_lt (c : Shape) (h : Fin c.groups) (i : Fin c.rows) :
    rowOffset c h i < c.groups * c.rows := by
  have hh := h.isLt
  have hi := i.isLt
  have hm := Nat.mul_le_mul_right c.rows (Nat.succ_le_of_lt hh)
  simp only [rowOffset]
  nlinarith

theorem offset_lt (c : Shape) (p : Coord c) : offset c p < c.elements := by
  have hr := rowOffset_lt c p.1 p.2.1
  have hd := p.2.2.isLt
  have hm := Nat.mul_le_mul_right c.dim (Nat.succ_le_of_lt hr)
  simp only [offset,Shape.elements,rowOffset] at *
  nlinarith

private theorem flatten_div (a d e : Nat) (hd : 0 < d) (he : e < d) :
    (a * d + e) / d = a := by
  rw [Nat.mul_comm a d,Nat.mul_add_div hd,Nat.div_eq_of_lt he,Nat.add_zero]

theorem offset_injective (c : Shape) : Function.Injective (offset c) := by
  intro a b hab
  have hda := a.2.2.isLt
  have hdb := b.2.2.isLt
  have hra := a.2.1.isLt
  have hrb := b.2.1.isLt
  have hd : a.2.2 = b.2.2 := by
    apply Fin.ext
    have hm := congrArg (· % c.dim) hab
    simpa only [offset,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hda,
      Nat.mod_eq_of_lt hdb] using hm
  have hr : a.1.val * c.rows + a.2.1.val = b.1.val * c.rows + b.2.1.val := by
    have hh := congrArg (· / c.dim) hab
    simpa only [offset,flatten_div _ _ _ c.dim_pos hda,
      flatten_div _ _ _ c.dim_pos hdb] using hh
  have hi : a.2.1 = b.2.1 := by
    apply Fin.ext
    have hh := congrArg (· % c.rows) hr
    simpa only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hra,
      Nat.mod_eq_of_lt hrb] using hh
  have hg : a.1 = b.1 := by
    apply Fin.ext
    have hh := congrArg (· / c.rows) hr
    simpa only [flatten_div _ _ _ c.rows_pos hra,
      flatten_div _ _ _ c.rows_pos hrb] using hh
  exact Prod.ext hg (Prod.ext hi hd)

/-- Canonical contiguous strides make the backward batch/head decoding agree
with the forward flattened head base, including nonzero batch/head IDs. -/
theorem backward_base_eq_forward (c : Shape) (group : Nat) :
    (group / c.heads) * (c.heads * c.rows * c.dim) +
      (group % c.heads) * (c.rows * c.dim) = group * (c.rows * c.dim) := by
  calc
    _ = (group / c.heads * c.heads + group % c.heads) * (c.rows * c.dim) := by ring
    _ = _ := by rw [Nat.div_add_mod']

/-- Every logical row has exactly one launched forward tile/lane. -/
theorem grid_coverage (c : Shape) (i : Fin c.rows) :
    i.val / 128 < c.blocks ∧ i.val % 128 < 128 ∧
    i.val / 128 * 128 + i.val % 128 = i.val := by
  have hi := i.isLt
  simp only [Shape.rows] at hi
  exact ⟨by omega,Nat.mod_lt _ (by decide),Nat.div_add_mod' _ _⟩

theorem grid_unique (b lane b' lane' : Nat) (hl : lane < 128)
    (hl' : lane' < 128) (h : b * 128 + lane = b' * 128 + lane') :
    b = b' ∧ lane = lane' := by omega

/-- The backward triangular loop visits every causally visible pair. -/
theorem backward_pair_coverage (i j : Nat) (h : j ≤ i) : j / 128 ≤ i / 128 := by omega

/-- The signed reset used between backward outer iterations returns Q/DO/DQ
from row N to row lo+128. This is integer arithmetic, avoiding Nat subtraction. -/
theorem pointer_reset (blocks keyBlock : Nat) :
    (blocks : Int) * 128 + ((keyBlock : Int) * 128 + (1 - blocks) * 128) =
      (keyBlock + 1 : Int) * 128 := by ring

/-- Source row/element offsets and four-byte FP32 displacements fit their
machine widths; the unused final DQ advance is also signed64 representable. -/
theorem machine_offsets (c : Shape) (p : Coord c) :
    offset c p < 2147483647 ∧
    4 * offset c p < 9223372036854775807 ∧
    4 * (c.elements + 16384) < 9223372036854775807 := by
  have h := offset_lt c p
  have he : c.elements ≤ 2147483647 := c.element_bound
  omega

#axiomsClean offset_injective
#axiomsClean backward_base_eq_forward
#axiomsClean grid_coverage
#axiomsClean pointer_reset
#axiomsClean machine_offsets
end VeriTile.Bench.Repairs.TritonAttention.Whole001
