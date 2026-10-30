import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardSnapshots
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RealCongruence

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

theorem body_logical_dx (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (i : Fin r.preparation.data.dx.count) :
    Storage.realLogical (r.realFinal m) r.output.1 i.val=
      RealProof.gradientSpec c.columns (inputX c m (i.val/c.columns)) (inputDY c dy m (i.val/c.columns))
        (inputW c m) (inputStats c m (i.val/c.columns)).mean (inputStats c m (i.val/c.columns)).inverse (i.val%c.columns) := by
  let layout := route_layout r.preparation saved
  have hc : i.val%c.columns<c.columns := Nat.mod_lt _ saved.positive
  have hr : i.val/c.columns<c.rows := by
    apply (Nat.div_lt_iff_lt_mul saved.positive).mpr
    rw [←saved.xCount,←layout.dxCount]
    exact i.isLt
  rw [body_real_dx,dx_saved_stats r m present _ hr]
  apply RealProof.gradient_congr _ _ _ _ _ _ _ _ _ _ hc
  · intro col hcol
    have hi := (row_offset_safe c.x c.rows c.columns _ col saved.xCount hr hcol).1
    exact dx_input_real r m c.x (present _ (by simp)) ⟨_,hi⟩
  · intro col hcol
    have hi := (row_offset_safe r.preparation.data.dy c.rows c.columns _ col layout.dyCount hr hcol).1
    exact dx_dy_real r m ⟨_,hi⟩
  · intro col hcol
    exact dx_input_real r m c.w (present _ (by simp)) ⟨col,lt_of_lt_of_le hcol saved.wCount⟩

theorem body_logical_dw (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (members : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → t.tensor∈inputs)
    (i : Fin r.preparation.data.dw.count) :
    Storage.realLogical (r.realFinal m) r.output.2.1 i.val=
      RealProof.weightGradientSpec c.columns c.rows (inputX c m) (inputDY c dy m) (inputStats c m) i.val := by
  rw [body_real_dw]
  unfold RealProof.weightGradientSpec
  by_cases hc : i.val<c.columns
  · simp only [hc,if_true]
    apply Finset.sum_congr rfl
    intro row hrow
    have hr : row<c.rows := Finset.mem_range.mp hrow
    have hi := (row_offset_safe c.x c.rows c.columns row i.val saved.xCount hr hc).1
    have hdy := (row_offset_safe r.preparation.data.dy c.rows c.columns row i.val
      (route_layout r.preparation saved).dyCount hr hc).1
    have hx := affine_input_real r m c.x (present _ (by simp)) (members _ (by simp)) rfl ⟨_,hi⟩
    have hy := affine_dy_real r m ⟨_,hdy⟩
    have hs := affine_saved_stats r m present members row hr
    change rows c.x c.columns (r.affineInput m) Memory.realAPI row i.val=inputX c m row i.val at hx
    change rows r.preparation.data.dy c.columns (r.affineInput m) Memory.realAPI row i.val=inputDY c dy m row i.val at hy
    rw [hx,hy,hs]
  · simp only [hc,if_false]

theorem body_logical_db (r : BodyRoute lib service c saved dy initial) (db : Dense) (out : Storage.Tensor)
    (hdb : r.preparation.data.db=some db) (hout : r.output.2.2.1=some out)
    (m : Storage.World (Nat → ℝ)) (i : Fin db.count) :
    Storage.realLogical (r.realFinal m) out i.val=
      RealProof.biasGradientSpec c.columns c.rows (inputDY c dy m) i.val := by
  rw [body_real_db r db out hdb hout]
  unfold RealProof.biasGradientSpec
  by_cases hc : i.val<c.columns
  · simp only [hc,if_true]
    apply Finset.sum_congr rfl
    intro row hrow
    have hr : row<c.rows := Finset.mem_range.mp hrow
    have hi := (row_offset_safe r.preparation.data.dy c.rows c.columns row i.val
      (route_layout r.preparation saved).dyCount hr hc).1
    exact affine_dy_real r m ⟨_,hi⟩
  · simp only [hc,if_false]

#axiomsClean body_logical_dx
#axiomsClean body_logical_dw
#axiomsClean body_logical_db
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
