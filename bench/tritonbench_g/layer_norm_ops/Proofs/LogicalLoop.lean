import bench.tritonbench_g.layer_norm_ops.Proofs.LogicalBackward
import bench.tritonbench_g.layer_norm_liger.Proofs.FoldFacts
namespace VeriTile.Bench.Repairs.LayerNormOps.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

def rowDW (p : BackwardParameters) (s : State ℝ) (row i : Nat) :=
  (s.mem p.dy (row*p.dyStride+i)).read .fp32 *
    (((s.mem p.x (row*p.xStride+i)).read .fp32 -
      if p.rms then 0 else (s.mem p.mean (row*p.meanStride)).read .fp32) *
      (s.mem p.rstd (row*p.rstdStride)).read .fp32)
def rowDB (p : BackwardParameters) (s : State ℝ) (row i : Nat) :=
  (s.mem p.dy (row*p.dyStride+i)).read .fp32

def Inputs (p : BackwardParameters) (s : State ℝ) (a : Backward.Acc ℝ p) :=
  ∀ r ∈ [p.x,p.dy,p.mean,p.rstd], a.state.mem r=s.mem r

def InputsFresh (p : BackwardParameters) :=
  ∀ r ∈ [p.x,p.dy,p.mean,p.rstd], Backward.RegionOutside p r

theorem inputs_update (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (hf : InputsFresh p) (row : Nat) (a : Backward.Acc ℝ p) (ha : Inputs p s a) :
    Inputs p s (Backward.update (algebra rest) p row a) := by
  intro r hr
  funext o
  rw [Backward.update_frame _ _ _ _ _ _ (hf r hr)]
  exact congrFun (ha r hr) o

theorem loop_dw (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (hf : InputsFresh p) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (algebra rest) p s).dw i =
      ∑ row ∈ Finset.Ico (Backward.start p s) (Backward.stop p s), rowDW p s row i.1.val := by
  have step : ∀ row (a : Backward.Acc ℝ p), Inputs p s a →
      (Backward.update (algebra rest) p row a).dw i = a.dw i+rowDW p s row i.1.val := by
    intro row a ha
    rw [backward_dw_step rest p a row i hi]
    simp [bdy,bhat,bx,bmu,binv,rowDW,ha p.dy (by simp),ha p.x (by simp),
      ha p.mean (by simp),ha p.rstd (by simp)]
  have result := Folds.project (Backward.update (algebra rest) p)
    (fun row v => v+rowDW p s row i.1.val) (Inputs p s) (fun a => a.dw i)
    (inputs_update rest p s hf) step (Backward.start p s) (Backward.stop p s)
    (Backward.initial (algebra rest) p s) (by intro r hr; rfl)
  dsimp only at result
  unfold Backward.accumulated
  rw [result,Folds.add_sum]
  simp [Backward.initial,Forward.literal,algebra,LayerNormLiger.Logical.algebra]

theorem loop_db (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (hf : InputsFresh p) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) (hb : p.hasBias=Bool.true) :
    (Backward.accumulated (algebra rest) p s).db i =
      ∑ row ∈ Finset.Ico (Backward.start p s) (Backward.stop p s), rowDB p s row i.1.val := by
  have step : ∀ row (a : Backward.Acc ℝ p), Inputs p s a →
      (Backward.update (algebra rest) p row a).db i = a.db i+rowDB p s row i.1.val := by
    intro row a ha
    rw [backward_db_step rest p a row i hi hb]
    simp [bdy,rowDB,ha p.dy (by simp)]
  have result := Folds.project (Backward.update (algebra rest) p)
    (fun row v => v+rowDB p s row i.1.val) (Inputs p s) (fun a => a.db i)
    (inputs_update rest p s hf) step (Backward.start p s) (Backward.stop p s)
    (Backward.initial (algebra rest) p s) (by intro r hr; rfl)
  dsimp only at result
  unfold Backward.accumulated
  rw [result,Folds.add_sum]
  simp [Backward.initial,Forward.literal,algebra,LayerNormLiger.Logical.algebra]

#axiomsClean inputs_update
#axiomsClean loop_dw
#axiomsClean loop_db
end
end VeriTile.Bench.Repairs.LayerNormOps.Logical
