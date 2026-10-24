import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardCall

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : Service platform api neg inputs Metadata}
  {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata} {identity : Nat}
  {c : SavedContext} {b : Dense} {saved : Saved c b} {dy : Storage.Tensor} {initial : State Store Metadata}

theorem dy_count (r : Call lib extra service identity c b saved dy initial) : dy.view.shape.prod=c.rows*c.columns :=
  (Preparation.materialize_metadata r.body.preparation.base.dyr).2.1.symm.trans r.body.ready.base.dyCount

theorem row_index (row col : Nat) (hr : row<c.rows) (hc : col<c.columns) : row*c.columns+col<c.rows*c.columns := by
  have h := Nat.mul_le_mul_right c.columns (show row+1≤c.rows by omega)
  nlinarith

theorem call_stats_transport (r : Call lib extra service identity c b saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (m : Storage.World (Nat → ℝ)) (row : Nat) (hr : row<c.rows) :
    inputStats c (r.preTransition.shadow m) row=inputStats c m row := by
  apply congrArg₂ VeriTile.Bench.Repairs.FastLayernorm.Statistics.mk
  · exact r.preTransition.realValues m c.mu.tensor (present _ (by simp)) row
      (by rw [←c.mu.countEq,saved.muCount]; exact hr)
  · exact r.preTransition.realValues m c.r.tensor (present _ (by simp)) row
      (by rw [←c.r.countEq,saved.rCount]; exact hr)

theorem call_x_transport (r : Call lib extra service identity c b saved dy initial) (present : initial.live c.x.tensor)
    (m : Storage.World (Nat → ℝ)) (row col : Nat) (hr : row<c.rows) (hc : col<c.columns) :
    inputX c (r.preTransition.shadow m) row col=inputX c m row col :=
  r.preTransition.realValues m c.x.tensor present _ (by rw [←c.x.countEq,saved.xCount]; exact row_index row col hr hc)

theorem call_dy_transport (r : Call lib extra service identity c b saved dy initial) (present : initial.live dy)
    (m : Storage.World (Nat → ℝ)) (row col : Nat) (hr : row<c.rows) (hc : col<c.columns) :
    inputDY c dy (r.preTransition.shadow m) row col=inputDY c dy m row col :=
  r.preTransition.realValues m dy present _ (by rw [dy_count r]; exact row_index row col hr hc)

theorem returned_dx_real (r : Call lib extra service identity c b saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (dyPresent : initial.live dy) (m : Storage.World (Nat → ℝ)) (i : Fin r.body.preparation.data.base.dx.count) :
    Storage.realLogical (r.realFinal m) r.body.returns.dx.tensor i.val=
      RealProof.Spec.gradient c.columns (inputX c m (i.val/c.columns)) (inputDY c dy m (i.val/c.columns))
        (inputW c m) (inputStats c m (i.val/c.columns)).mean (inputStats c m (i.val/c.columns)).inverse (i.val%c.columns) := by
  apply (post_real r m 0 i).trans
  dsimp only [ReturnRoute.outputs]
  rw [body_dx_real r.body (accessed_live r)]
  have hi : i.val<c.rows*c.columns := by simpa only [←r.body.ready.base.dxCount] using i.isLt
  have hn : 0<c.columns := Nat.pos_of_mul_pos_left (lt_of_le_of_lt (Nat.zero_le _) hi)
  have hr := (Nat.div_lt_iff_lt_mul hn).mpr hi
  have hc := Nat.mod_lt i.val hn
  rw [call_stats_transport r present m _ hr]
  apply VeriTile.Bench.Repairs.FastLayernorm.RealProof.gradient_congr _ _ _ _ _ _ _ _ _ _ hc
  · exact fun col hc => call_x_transport r (present _ (by simp)) m _ col hr hc
  · exact fun col hc => call_dy_transport r dyPresent m _ col hr hc
  · intro col hc
    exact r.preTransition.realValues m c.w.tensor (present _ (by simp)) col
      (by rw [←c.w.countEq,saved.wCount]; exact hc)

theorem returned_dw_real (r : Call lib extra service identity c b saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (dyPresent : initial.live dy) (m : Storage.World (Nat → ℝ)) (i : Fin r.body.preparation.data.base.dw.count) :
    Storage.realLogical (r.realFinal m) r.body.returns.wr.output i.val=
      RealProof.Spec.weightGradient c.columns c.rows (inputX c m) (inputDY c dy m) (inputStats c m) i.val := by
  apply (post_real r m 1 i).trans
  dsimp only [ReturnRoute.outputs]
  rw [body_dw_real r.body (accessed_live r)]
  have hc : i.val<c.columns := by simpa only [r.body.ready.base.dwCount] using i.isLt
  simp only [RealProof.Spec.weightGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec,hc,if_true]
  apply Finset.sum_congr rfl
  intro row hrow
  have hr := Finset.mem_range.mp hrow
  rw [call_x_transport r (present _ (by simp)) m row i.val hr hc,call_dy_transport r dyPresent m row i.val hr hc,
    call_stats_transport r present m row hr]

theorem returned_db_real (r : Call lib extra service identity c b saved dy initial)
    (dyPresent : initial.live dy) (m : Storage.World (Nat → ℝ)) (i : Fin r.body.preparation.data.base.db.count) :
    Storage.realLogical (r.realFinal m) r.body.returns.br.output i.val=
      RealProof.Spec.biasGradient c.columns c.rows (inputDY c dy m) i.val := by
  apply (post_real r m 2 i).trans
  dsimp only [ReturnRoute.outputs]
  rw [body_db_real]
  have hc : i.val<c.columns := by simpa only [r.body.ready.base.dbCount] using i.isLt
  simp only [RealProof.Spec.biasGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.biasGradientSpec,hc,if_true]
  apply Finset.sum_congr rfl
  intro row hrow
  exact call_dy_transport r dyPresent m row i.val (Finset.mem_range.mp hrow) hc

#axiomsClean returned_dx_real
#axiomsClean returned_dw_real
#axiomsClean returned_db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
