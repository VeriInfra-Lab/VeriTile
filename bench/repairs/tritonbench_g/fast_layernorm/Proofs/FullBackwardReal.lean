import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardPost
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReturnShape

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {identity : Nat} {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

theorem stats_transport (tr : Transition api neg inputs before after) (c : SavedContext) (saved : SavedLayout c)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → before.live t.tensor)
    (m : Storage.World (Nat → ℝ)) (row : Nat) (hr : row<c.rows) :
    inputStats c (tr.shadow m) row=inputStats c m row := by
  apply congrArg₂ Statistics.mk
  · exact tr.realValues m c.mu.tensor (present _ (by simp)) row (by rw [←c.mu.countEq,saved.muCount]; exact hr)
  · exact tr.realValues m c.r.tensor (present _ (by simp)) row (by rw [←c.r.countEq,saved.rCount]; exact hr)

theorem dy_count (r : SuccessfulCall lib service identity c saved dy initial) : dy.view.shape.prod=c.rows*c.columns :=
  (Preparation.materialize_metadata r.body.preparation.dyr).2.1.symm.trans (route_layout r.body.preparation saved).dyCount

theorem dx_real (r : SuccessfulCall lib service identity c saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (dyPresent : initial.live dy) (m : Storage.World (Nat → ℝ)) (i : Fin r.body.preparation.data.dx.count) :
    Storage.realLogical (r.realFinal m) r.body.output.1 i.val=
      RealProof.gradientSpec c.columns (inputX c m (i.val/c.columns)) (inputDY c dy m (i.val/c.columns))
        (inputW c m) (inputStats c m (i.val/c.columns)).mean (inputStats c m (i.val/c.columns)).inverse (i.val%c.columns) := by
  let layout := route_layout r.body.preparation saved
  have hi : i.val<r.body.output.1.view.shape.prod := by
    rw [(dx_return_storage r.body).2,saved.inputShape,←layout.dxCount]
    exact i.isLt
  rw [post_real r _ (by simp [GradientReturn.returned]) m i.val hi]
  rw [body_logical_dx r.body (r.preTransition.shadow m) (accessed_live r)]
  have hc : i.val%c.columns<c.columns := Nat.mod_lt _ saved.positive
  have hr : i.val/c.columns<c.rows := by
    apply (Nat.div_lt_iff_lt_mul saved.positive).mpr
    rw [←saved.xCount,←layout.dxCount]
    exact i.isLt
  rw [stats_transport r.preTransition c saved present m _ hr]
  apply RealProof.gradient_congr _ _ _ _ _ _ _ _ _ _ hc
  · intro col hcol
    have h := (Execution.row_offset_safe c.x c.rows c.columns _ col saved.xCount hr hcol).1
    exact r.preTransition.realValues m c.x.tensor (present _ (by simp)) _ (by rw [←c.x.countEq]; exact h)
  · intro col hcol
    have h := (Execution.row_offset_safe c.x c.rows c.columns _ col saved.xCount hr hcol).1
    have hdy : (i.val/c.columns)*c.columns+col<dy.view.shape.prod := by rw [dy_count r,←saved.xCount]; exact h
    exact r.preTransition.realValues m dy dyPresent _ hdy
  · intro col hcol
    exact r.preTransition.realValues m c.w.tensor (present _ (by simp)) _
      (by rw [←c.w.countEq]; exact lt_of_lt_of_le hcol saved.wCount)

theorem dw_real (r : SuccessfulCall lib service identity c saved dy initial)
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (members : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → t.tensor∈inputs)
    (dyPresent : initial.live dy) (m : Storage.World (Nat → ℝ)) (i : Fin r.body.preparation.data.dw.count) :
    Storage.realLogical (r.realFinal m) r.body.output.2.1 i.val=
      RealProof.weightGradientSpec c.columns c.rows (inputX c m) (inputDY c dy m) (inputStats c m) i.val := by
  have hi : i.val<r.body.output.2.1.view.shape.prod := by
    rw [(dw_return_metadata r.body).1,saved.weightShape,←(route_layout r.body.preparation saved).dwCount]
    exact i.isLt
  rw [post_real r _ (by simp [GradientReturn.returned]) m i.val hi]
  rw [body_logical_dw r.body (r.preTransition.shadow m) (accessed_live r) members]
  unfold RealProof.weightGradientSpec
  by_cases hc : i.val<c.columns
  · simp only [hc,if_true]
    apply Finset.sum_congr rfl
    intro row hrow
    have hr := Finset.mem_range.mp hrow
    have h := (Execution.row_offset_safe c.x c.rows c.columns row i.val saved.xCount hr hc).1
    have hx := r.preTransition.realValues m c.x.tensor (present _ (by simp)) _ (by rw [←c.x.countEq]; exact h)
    have hdy : row*c.columns+i.val<dy.view.shape.prod := by rw [dy_count r,←saved.xCount]; exact h
    have hy := r.preTransition.realValues m dy dyPresent _ hdy
    change inputX c (r.preTransition.shadow m) row i.val=inputX c m row i.val at hx
    change inputDY c dy (r.preTransition.shadow m) row i.val=inputDY c dy m row i.val at hy
    rw [hx,hy,stats_transport r.preTransition c saved present m row hr]
  · simp only [hc,if_false]

theorem db_real (r : SuccessfulCall lib service identity c saved dy initial) (db : Dense) (out : Storage.Tensor)
    (hdb : r.body.preparation.data.db=some db) (hout : r.body.output.2.2.1=some out)
    (dyPresent : initial.live dy) (m : Storage.World (Nat → ℝ)) (i : Fin db.count) :
    Storage.realLogical (r.realFinal m) out i.val=
      RealProof.biasGradientSpec c.columns c.rows (inputDY c dy m) i.val := by
  have hi : i.val<out.view.shape.prod := by rw [(GradientReturn.db_metadata r.body.returns db out hdb hout).2.2]; exact i.isLt
  rw [post_real r out (by simp [GradientReturn.returned,hout]) m i.val hi]
  rw [body_logical_db r.body db out hdb hout]
  unfold RealProof.biasGradientSpec
  by_cases hc : i.val<c.columns
  · simp only [hc,if_true]
    apply Finset.sum_congr rfl
    intro row hrow
    have hr := Finset.mem_range.mp hrow
    have h := (Execution.row_offset_safe c.x c.rows c.columns row i.val saved.xCount hr hc).1
    have hdy : row*c.columns+i.val<dy.view.shape.prod := by rw [dy_count r,←saved.xCount]; exact h
    exact r.preTransition.realValues m dy dyPresent _ hdy
  · simp only [hc,if_false]

#axiomsClean dx_real
#axiomsClean dw_real
#axiomsClean db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
