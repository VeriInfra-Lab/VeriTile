import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Geometry

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001

/-- The host's one-dimensional preprocess grid is exactly the concatenation
of all forward batch/head grids. No last row or head is omitted. -/
theorem preprocess_grid (c : Shape) (h : Fin c.groups) (i : Fin c.rows) :
    (rowOffset c h i) / 128 = h.val * c.blocks + i.val / 128 ∧
    (rowOffset c h i) % 128 = i.val % 128 ∧
    h.val * c.blocks + i.val / 128 < c.groups * c.blocks := by
  have hc := grid_coverage c i
  have hh := h.isLt
  have hm := Nat.mul_le_mul_right c.blocks (Nat.succ_le_of_lt hh)
  have hoff : rowOffset c h i = 128 * (h.val * c.blocks) + i.val := by
    simp only [rowOffset,Shape.rows]
    ring
  rw [hoff]
  refine ⟨?_,?_,?_⟩
  · rw [Nat.mul_add_div (by decide : 0 < 128)]
  · simp
  · nlinarith

/-- The native forward upper bound cannot exceed N, and every loaded Q/K/V
lane is in bounds on the admitted complete-block domain. -/
theorem forward_block_bounds (c : Shape) (block : Fin c.blocks) (key : Nat)
    (hk : key ≤ block.val) (lane : Fin 128) :
    (block.val + 1) * 128 ≤ c.rows ∧
    block.val * 128 + lane.val < c.rows ∧
    key * 128 + lane.val < c.rows := by
  have hb := block.isLt
  have hl := lane.isLt
  simp only [Shape.rows]
  omega

/-- Both nested backward loops range over complete tensor rows. -/
theorem backward_block_bounds (c : Shape) (key query : Fin c.blocks)
    (_order : key.val ≤ query.val) (lane : Fin 128) :
    key.val * 128 + lane.val < c.rows ∧ query.val * 128 + lane.val < c.rows := by
  have hk := key.isLt
  have hq := query.isLt
  have hl := lane.isLt
  simp only [Shape.rows]
  omega

/-- The negative source reset is representable as signed32. -/
theorem backward_reset_bounds (c : Shape) (key : Fin c.blocks) :
    (128 : Int) - c.rows ≤ (key.val : Int) * 128 + (1 - c.blocks) * 128 ∧
    (key.val : Int) * 128 + (1 - c.blocks) * 128 ≤ 0 := by
  have hk := key.isLt
  simp only [Shape.rows,Nat.cast_mul,Nat.cast_ofNat]
  omega

/-- The normalizer/maximum/Delta row arrays are also bounded by the admitted
logical element count, so the existing element-count guard covers their indices. -/
theorem row_buffer_machine_bounds (c : Shape) (h : Fin c.groups) (i : Fin c.rows) :
    rowOffset c h i < 2147483647 := by
  have hi := rowOffset_lt c h i
  have hd := c.dim_pos
  have he : c.groups * c.rows * c.dim ≤ 2147483647 := c.element_bound
  have hm : c.groups * c.rows ≤ c.groups * c.rows * c.dim := by
    simpa using Nat.mul_le_mul_left (c.groups * c.rows) hd
  omega

#axiomsClean preprocess_grid
#axiomsClean forward_block_bounds
#axiomsClean backward_block_bounds
#axiomsClean backward_reset_bounds
#axiomsClean row_buffer_machine_bounds
end VeriTile.Bench.Repairs.TritonAttention.Whole001
