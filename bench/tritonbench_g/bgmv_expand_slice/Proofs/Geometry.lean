import bench.tritonbench_g.bgmv_expand_slice.Sources.Kernels

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Geometry
open VeriTile.Triton

structure Program (h : Call) where
  batch : Nat
  row : Nat
  batchInside : batch<h.batches
  rowInside : row<h.N

def inputOffset (h : Call) (batch k : Nat) := batch*h.BK+k
def weightOffset (h : Call) (slot row n k : Nat) :=
  slot*((h.N+255)*h.BK)+(row+n)*h.BK+k
def outputOffset (h : Call) (batch row n : Nat) :=
  batch*(h.columns+255)+row+h.slice+n

theorem input_inside (h : Call) (p : Program h) (k : Fin h.BK) :
    inputOffset h p.batch k.val<h.batches*h.BK := by
  unfold inputOffset
  have a := Nat.mul_le_mul_right h.BK (Nat.succ_le_of_lt p.batchInside)
  have b := k.isLt
  rw [Nat.succ_mul] at a
  omega

theorem weight_inside (h : Call) (p : Program h) (slot : Nat)
    (selected : slot<h.adapters) (n : Fin 256) (k : Fin h.BK) :
    weightOffset h slot p.row n.val k.val<h.adapters*((h.N+255)*h.BK) := by
  have row : p.row+n.val<h.N+255 := by have := p.rowInside; have := n.isLt; omega
  have r := Nat.mul_le_mul_right h.BK (Nat.succ_le_of_lt row)
  have a := Nat.mul_le_mul_right ((h.N+255)*h.BK) (Nat.succ_le_of_lt selected)
  have kbound := k.isLt
  unfold weightOffset
  rw [Nat.succ_mul] at a
  rw [Nat.succ_mul] at r
  omega

theorem output_inside (h : Call) (p : Program h) (n : Fin 256) :
    outputOffset h p.batch p.row n.val<h.batches*(h.columns+255) := by
  have bounds := h.sliceInside
  have row := p.rowInside
  have lane := n.isLt
  have a := Nat.mul_le_mul_right (h.columns+255) (Nat.succ_le_of_lt p.batchInside)
  unfold outputOffset
  rw [Nat.succ_mul] at a
  omega

theorem output_injective (h : Call) (p q : Program h)
    (same : outputOffset h p.batch p.row 0=outputOffset h q.batch q.row 0) :
    p.batch=q.batch ∧ p.row=q.row := by
  have width := h.sliceInside
  have pr := p.rowInside
  have qr := q.rowInside
  have rows : p.row+h.slice<h.columns+255 ∧ q.row+h.slice<h.columns+255 := by omega
  unfold outputOffset at same
  have b := congrArg (fun v => v/(h.columns+255)) same
  dsimp only at b
  have pb : (p.batch*(h.columns+255)+p.row+h.slice+0)/(h.columns+255)=p.batch := by
    rw [Nat.add_zero, Nat.add_assoc, Nat.add_comm (p.batch*(h.columns+255)), Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt rows.1]
    simp
  have qb : (q.batch*(h.columns+255)+q.row+h.slice+0)/(h.columns+255)=q.batch := by
    rw [Nat.add_zero, Nat.add_assoc, Nat.add_comm (q.batch*(h.columns+255)), Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt rows.2]
    simp
  rw [pb,qb] at b
  exact ⟨b,by simpa [b] using same⟩

#axiomsClean input_inside
#axiomsClean weight_inside
#axiomsClean output_inside
#axiomsClean output_injective
end VeriTile.Bench.Repairs.BgmvExpandSlice.Geometry
