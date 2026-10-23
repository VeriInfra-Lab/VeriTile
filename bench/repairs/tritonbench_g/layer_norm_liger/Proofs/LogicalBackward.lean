import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Logical
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def bx (p : Backward.Parameters) (a : Backward.Acc ℝ p) (row i : Nat) :=
  (a.state.mem p.x (row*p.xStride+i)).read .fp32
def bw (p : Backward.Parameters) (a : Backward.Acc ℝ p) (i : Nat) := (a.state.mem p.weight i).read .fp32
def bdy (p : Backward.Parameters) (a : Backward.Acc ℝ p) (row i : Nat) :=
  (a.state.mem p.dy (row*p.dyStride+i)).read .fp32

theorem backward_xhat (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) :
    Backward.xhat (algebra rest) p a row i = if i.1.val < p.columns then
      xhat (bx p a row) (Backward.mean p a row) (Backward.rstd p a row) i.1.val else 0 := by
  by_cases hi : i.1.val < p.columns <;>
    simp [Backward.xhat, Backward.load, Backward.cast, algebra, bx, xhat, hi]

theorem backward_wdy (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) :
    Backward.wdy (algebra rest) p a row i = if i.1.val < p.columns then
      bw p a i.1.val*bdy p a row i.1.val else 0 := by
  by_cases hi : i.1.val < p.columns <;>
    simp [Backward.wdy, Backward.load, Backward.padding, algebra, bw, bdy, hi]

theorem backward_c1 (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (hb : p.columns ≤ p.block) :
    Backward.c1 (algebra rest) p a row =
      sum p.columns (fun i => xhat (bx p a row) (Backward.mean p a row) (Backward.rstd p a row) i *
        (bw p a i*bdy p a row i)) / p.columns := by
  change ((∑ i : Fin p.block, Backward.xhat (algebra rest) p a row (i,PUnit.unit) *
    Backward.wdy (algebra rest) p a row (i,PUnit.unit)) / p.columns) = _
  simp_rw [backward_xhat, backward_wdy]
  have he : (fun i : Fin p.block =>
      (if i.val < p.columns then xhat (bx p a row) (Backward.mean p a row) (Backward.rstd p a row) i.val else 0) *
      (if i.val < p.columns then bw p a i.val*bdy p a row i.val else 0)) =
      (fun i : Fin p.block => if i.val < p.columns then
        xhat (bx p a row) (Backward.mean p a row) (Backward.rstd p a row) i.val *
        (bw p a i.val*bdy p a row i.val) else 0) := by
    funext i
    split_ifs <;> simp
  rw [he, padded_sum _ _ hb (fun i => xhat (bx p a row) (Backward.mean p a row) (Backward.rstd p a row) i * (bw p a i*bdy p a row i))]

theorem backward_c2 (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (hb : p.columns ≤ p.block) :
    Backward.c2 (algebra rest) sumPrimitive p a row =
      sum p.columns (fun i => bw p a i*bdy p a row i) / p.columns := by
  change ((∑ i : Fin p.block, Backward.wdy (algebra rest) p a row (i,PUnit.unit)) / p.columns) = _
  simp_rw [backward_wdy]
  rw [padded_sum _ _ hb (fun i => bw p a i*bdy p a row i)]

theorem backward_dx (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (hb : p.columns ≤ p.block) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    Backward.dxValue (algebra rest) sumPrimitive p a row i =
      dx p.columns (bx p a row) (bw p a) (bdy p a row) (Backward.mean p a row) (Backward.rstd p a row) i.1.val := by
  unfold Backward.dxValue
  rw [backward_xhat, backward_wdy, backward_c1 rest p a row hb, backward_c2 rest p a row hb]
  simp [hi, Backward.binary, Backward.cast, algebra, dx]

theorem backward_dw_step (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (Backward.update (algebra rest) sumPrimitive p row a).dw i = a.dw i +
      bdy p a row i.1.val*xhat (bx p a row) (Backward.mean p a row) (Backward.rstd p a row) i.1.val := by
  simp only [Backward.update, backward_xhat]
  simp [Backward.binary, Backward.cast, Backward.load, algebra, bdy, hi]

theorem backward_db_step (rest : Algebra ℝ) (p : Backward.Parameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (Backward.update (algebra rest) sumPrimitive p row a).db i = a.db i + bdy p a row i.1.val := by
  simp [Backward.update, Backward.binary, Backward.cast, Backward.load, algebra, bdy, hi]

#axiomsClean backward_xhat
#axiomsClean backward_wdy
#axiomsClean backward_c1
#axiomsClean backward_c2
#axiomsClean backward_dx
#axiomsClean backward_dw_step
#axiomsClean backward_db_step
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
