import bench.tritonbench_g.layer_norm_ops.Proofs.Logical
import bench.tritonbench_g.layer_norm_ops.Proofs.Readback
namespace VeriTile.Bench.Repairs.LayerNormOps.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger.Logical (sum padded_sum)
open scoped BigOperators
noncomputable section
set_option linter.unusedSimpArgs false

def bx (p : BackwardParameters) (a : Backward.Acc ℝ p) (row i : Nat) :=
  (a.state.mem p.x (row*p.xStride+i)).read .fp32
def bdy (p : BackwardParameters) (a : Backward.Acc ℝ p) (row i : Nat) :=
  (a.state.mem p.dy (row*p.dyStride+i)).read .fp32
def bmu (p : BackwardParameters) (a : Backward.Acc ℝ p) (row : Nat) :=
  (a.state.mem p.mean (row*p.meanStride)).read .fp32
def binv (p : BackwardParameters) (a : Backward.Acc ℝ p) (row : Nat) :=
  (a.state.mem p.rstd (row*p.rstdStride)).read .fp32
def bhat (p : BackwardParameters) (a : Backward.Acc ℝ p) (row i : Nat) :=
  (bx p a row i - if p.rms then 0 else bmu p a row)*binv p a row

theorem backward_xhat (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) :
    Backward.xhat (algebra rest) p a row i = if i.1.val < p.columns then bhat p a row i.1.val else 0 := by
  cases hr : p.rms <;> by_cases hi : i.1.val < p.columns <;>
    simp [Backward.xhat,Backward.loaded,Backward.meanValue,Forward.cast,Forward.unhintedBinary,
      algebra,LayerNormLiger.Logical.algebra,bhat,bx,bmu,binv,hr,hi]

theorem backward_wdy (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (w : Nat → ℝ) (hw : ∀ i : TileIndex [p.block], i.1.val < p.columns → a.weight i=w i.1.val)
    (i : TileIndex [p.block]) :
    Backward.wdy (algebra rest) p a row i = if i.1.val < p.columns then w i.1.val*bdy p a row i.1.val else 0 := by
  by_cases hi : i.1.val < p.columns
  · simp [Backward.wdy,Backward.dy,Backward.loaded,Forward.cast,Forward.binary,
      algebra,LayerNormLiger.Logical.algebra,hi,hw i hi,bdy]
  · simp [Backward.wdy,Backward.dy,Backward.loaded,Forward.cast,Forward.binary,Forward.padding,Forward.literal,
      algebra,LayerNormLiger.Logical.algebra,hi]

theorem backward_c1 (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (w : Nat → ℝ) (hw : ∀ i : TileIndex [p.block], i.1.val < p.columns → a.weight i=w i.1.val)
    (hb : p.columns ≤ p.block) :
    Backward.c1 (algebra rest) p a row = sum p.columns (fun i => bhat p a row i*(w i*bdy p a row i))/p.columns := by
  change (∑ i : Fin p.block, Backward.xhat (algebra rest) p a row (i,PUnit.unit)*
    Backward.wdy (algebra rest) p a row (i,PUnit.unit))/p.columns = _
  simp_rw [backward_xhat,backward_wdy rest p a row w hw]
  have he : (fun i : Fin p.block =>
      (if i.val<p.columns then bhat p a row i.val else 0)*(if i.val<p.columns then w i.val*bdy p a row i.val else 0)) =
      fun i => if i.val<p.columns then bhat p a row i.val*(w i.val*bdy p a row i.val) else 0 := by
    funext i; split_ifs <;> simp
  rw [he,padded_sum _ _ hb (fun i => bhat p a row i*(w i*bdy p a row i))]

theorem backward_c2 (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (w : Nat → ℝ) (hw : ∀ i : TileIndex [p.block], i.1.val < p.columns → a.weight i=w i.1.val)
    (hb : p.columns ≤ p.block) :
    Backward.c2 (algebra rest) p a row = sum p.columns (fun i => w i*bdy p a row i)/p.columns := by
  change (∑ i : Fin p.block, Backward.wdy (algebra rest) p a row (i,PUnit.unit))/p.columns = _
  simp_rw [backward_wdy rest p a row w hw]
  rw [padded_sum _ _ hb (fun i => w i*bdy p a row i)]

theorem backward_dxBase (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (w : Nat → ℝ) (hw : ∀ i : TileIndex [p.block], i.1.val < p.columns → a.weight i=w i.1.val)
    (hb : p.columns ≤ p.block) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    Backward.dxBase (algebra rest) p a row i =
      dx p.columns p.rms Bool.false (bx p a row) w (bdy p a row) (fun _ => 0) (bmu p a row) (binv p a row) i.1.val := by
  unfold Backward.dxBase
  rw [backward_xhat,backward_wdy rest p a row w hw,backward_c1 rest p a row w hw hb]
  cases hr : p.rms
  · rw [backward_c2 rest p a row w hw hb]
    simp [hr,hi,Backward.rstdValue,Forward.cast,Forward.binary,algebra,LayerNormLiger.Logical.algebra,
      dx,bhat,binv]
    ring_nf
    exact Or.inl True.intro
  · simp [hr,hi,Backward.rstdValue,Forward.cast,Forward.binary,algebra,LayerNormLiger.Logical.algebra,
      dx,bhat,binv]

theorem backward_dw_step (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (Backward.update (algebra rest) p row a).dw i = a.dw i+bdy p a row i.1.val*bhat p a row i.1.val := by
  simp only [Backward.update,backward_xhat]
  simp [Backward.dy,Backward.loaded,Forward.binary,Forward.cast,algebra,LayerNormLiger.Logical.algebra,bdy,hi]

theorem backward_db_step (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) (hb : p.hasBias = Bool.true) :
    (Backward.update (algebra rest) p row a).db i = a.db i+bdy p a row i.1.val := by
  simp [Backward.update,Backward.dy,Backward.loaded,Forward.binary,Forward.cast,
    algebra,LayerNormLiger.Logical.algebra,bdy,hi,hb]

#axiomsClean backward_xhat
#axiomsClean backward_wdy
#axiomsClean backward_c1
#axiomsClean backward_c2
#axiomsClean backward_dxBase
#axiomsClean backward_dw_step
#axiomsClean backward_db_step
end
end VeriTile.Bench.Repairs.LayerNormOps.Logical
