import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardBodyValues

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : Service platform api neg inputs Metadata}
  {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata}
  {c : SavedContext} {b : Dense} {saved : Saved c b} {dy : Storage.Tensor} {initial : State Store Metadata}

def inputX (c : SavedContext) (m : Storage.World (Nat → ℝ)) (row col : Nat) :=
  Storage.realLogical m c.x.tensor (row*c.columns+col)
def inputDY (c : SavedContext) (dy : Storage.Tensor) (m : Storage.World (Nat → ℝ)) (row col : Nat) :=
  Storage.realLogical m dy (row*c.columns+col)
def inputW (c : SavedContext) (m : Storage.World (Nat → ℝ)) := Storage.realLogical m c.w.tensor
def inputStats (c : SavedContext) (m : Storage.World (Nat → ℝ)) (row : Nat) : Reuse.Statistics ℝ :=
  ⟨Storage.realLogical m c.mu.tensor row,Storage.realLogical m c.r.tensor row⟩

theorem body_dense_real (r : BodyRoute lib extra service c b saved dy initial)
    (m : Storage.World (Nat → ℝ)) (t : Dense) (present : initial.live t.tensor) (i : Nat) (hi : i<t.count) :
    Execution.E.vector t (r.preReal m) Memory.realAPI i=Storage.realLogical m t.tensor i := by
  have hic : i<t.tensor.view.shape.prod := by rw [←t.countEq]; exact hi
  have h := launch_pre_real r.launch (r.preparation.transition.shadow m) t.tensor
    (r.preparation.transition.live _ present) i hic
  rw [dense_real _ _ _ hi] at h
  exact h.trans (r.preparation.transition.realValues m t.tensor present i hic)

theorem body_dy_real (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (row col : Nat) (hr : row<c.rows) (hc : col<c.columns) :
    Execution.E.rows r.preparation.data.base.dy c.columns (r.preReal m) Memory.realAPI row col=inputDY c dy m row col := by
  have hi : row*c.columns+col<r.preparation.data.base.dy.count := by
    rw [r.ready.base.dyCount]
    have hmul := Nat.mul_le_mul_right c.columns (show row+1≤c.rows by omega)
    nlinarith
  have h := launch_pre_real r.launch (r.preparation.transition.shadow m) r.preparation.data.base.dy.tensor
    (preparation_present r.preparation).1 _ (by rw [←r.preparation.data.base.dy.countEq]; exact hi)
  rw [dense_real _ _ _ hi] at h
  exact h.trans (preparation_dy_real r.preparation m ⟨_,hi⟩)

theorem body_x_real (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (present : initial.live c.x.tensor) (row col : Nat) (hr : row<c.rows) (hc : col<c.columns) :
    Execution.E.rows c.x c.columns (r.preReal m) Memory.realAPI row col=inputX c m row col := by
  apply body_dense_real r m c.x present
  rw [saved.xCount]
  have hmul := Nat.mul_le_mul_right c.columns (show row+1≤c.rows by omega)
  nlinarith

theorem body_stats_real (r : BodyRoute lib extra service c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (present : initial.live c.mu.tensor ∧ initial.live c.r.tensor) (row : Nat) (hr : row<c.rows) :
    Execution.E.saved Memory.realAPI c.r c.mu (r.preReal m) row=inputStats c m row := by
  apply congrArg₂ VeriTile.Bench.Repairs.FastLayernorm.Statistics.mk
  · exact body_dense_real r m c.mu present.1 row (by simpa only [saved.muCount] using hr)
  · exact body_dense_real r m c.r present.2 row (by simpa only [saved.rCount] using hr)

theorem body_dx_real (r : BodyRoute lib extra service c b saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Fin r.preparation.data.base.dx.count) :
    Storage.realLogical (r.realFinal m) r.returns.dx.tensor i.val=
      RealProof.Spec.gradient c.columns (inputX c m (i.val/c.columns)) (inputDY c dy m (i.val/c.columns))
        (inputW c m) (inputStats c m (i.val/c.columns)).mean (inputStats c m (i.val/c.columns)).inverse (i.val%c.columns) := by
  have hi : i.val<c.rows*c.columns := by simpa only [←r.ready.base.dxCount] using i.isLt
  have hn : 0<c.columns := Nat.pos_of_mul_pos_left (lt_of_le_of_lt (Nat.zero_le _) hi)
  have hr := (Nat.div_lt_iff_lt_mul hn).mpr hi
  have hc := Nat.mod_lt i.val hn
  rw [body_dx_snapshot r m i,body_stats_real r m ⟨present _ (by simp),present _ (by simp)⟩ _ hr]
  apply VeriTile.Bench.Repairs.FastLayernorm.RealProof.gradient_congr _ _ _ _ _ _ _ _ _ _ hc
  · exact fun col hc => body_x_real r m (present _ (by simp)) _ col hr hc
  · exact fun col hc => body_dy_real r m _ col hr hc
  · exact fun col hc => body_dense_real r m c.w (present _ (by simp)) col (by simpa only [saved.wCount] using hc)

theorem body_dw_real (r : BodyRoute lib extra service c b saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Fin r.preparation.data.base.dw.count) :
    Storage.realLogical (r.realFinal m) r.returns.wr.output i.val=
      RealProof.Spec.weightGradient c.columns c.rows (inputX c m) (inputDY c dy m) (inputStats c m) i.val := by
  rw [body_dw_snapshot r m i]
  have hc : i.val<c.columns := by simpa only [r.ready.base.dwCount] using i.isLt
  simp only [RealProof.Spec.weightGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec,hc,if_true]
  apply Finset.sum_congr rfl
  intro row hrow
  have hr := Finset.mem_range.mp hrow
  rw [body_x_real r m (present _ (by simp)) row i.val hr hc,body_dy_real r m row i.val hr hc,
    body_stats_real r m ⟨present _ (by simp),present _ (by simp)⟩ row hr]

theorem body_db_real (r : BodyRoute lib extra service c b saved dy initial)
    (m : Storage.World (Nat → ℝ)) (i : Fin r.preparation.data.base.db.count) :
    Storage.realLogical (r.realFinal m) r.returns.br.output i.val=
      RealProof.Spec.biasGradient c.columns c.rows (inputDY c dy m) i.val := by
  rw [body_db_snapshot r m i]
  have hc : i.val<c.columns := by simpa only [r.ready.base.dbCount] using i.isLt
  simp only [RealProof.Spec.biasGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.biasGradientSpec,hc,if_true]
  apply Finset.sum_congr rfl
  intro row hrow
  exact body_dy_real r m row i.val (Finset.mem_range.mp hrow) hc

#axiomsClean body_dx_real
#axiomsClean body_dw_real
#axiomsClean body_db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
