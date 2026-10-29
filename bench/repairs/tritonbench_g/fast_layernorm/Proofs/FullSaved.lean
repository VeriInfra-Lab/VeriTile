import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SavedForward
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.FullForwardReal

namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper Execution
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform} {block warps : Nat}
  {lib : APIHost api neg inputs Metadata} {service : Runtime.Service platform api neg inputs Metadata}
  {identity : Nat} {x w : Storage.Tensor} {bias : Option Storage.Tensor} {shape : ShapeInfo x}
  {checks : Checks x w bias shape block warps} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

theorem body_real_input (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (t : Dense) (member : t∈reads r.preparation.data) (i : Fin t.count) :
    Storage.realLogical (r.realFinal m epsReal) t.tensor i.val=Storage.realLogical (r.preReal m) t.tensor i.val := by
  let layout := Layout.route_layout r.preparation shape.positive (shape_count shape) checks.weight checks.bias
  have live := r.dispatch.preTransition.live _ ((route_present r.preparation).1 t member)
  have htail := r.tail.realValues (r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal)
    t.tensor live i.val (by rw [←t.countEq]; exact i.isLt)
  have hk := trace_prefix_input Memory.realAPI
    (forwardTrace RealProof.ops r.dispatch.code.plan Memory.realAPI r.preparation.data layout epsReal (r.preReal m))
    (r.preReal m) t (layout.inputsOwned t member)
    (fun k => by rw [Runtime.forward_buffer]; exact layout.freshInputs k t member)
    _ (scheduled_subset _ r.dispatch.schedule) i
  rw [Storage.dense_real _ _ _ i.isLt,Storage.dense_real _ _ _ i.isLt] at htail ⊢
  exact htail.trans hk

theorem body_saved_x_real (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin x.view.shape.prod) :
    Storage.realLogical (r.realFinal m epsReal) r.context.x.tensor i.val=Storage.realLogical m x i.val := by
  have hi : i.val<r.preparation.data.x.count := by change i.val<r.preparation.xr.output.count; rw [(materialize_metadata r.preparation.xr).2.1]; exact i.isLt
  have hk := body_real_input r m epsReal r.preparation.data.x (by simp [reads]) ⟨i.val,hi⟩
  have ht := r.dispatch.preTransition.realValues (r.preparation.transition.shadow m) r.preparation.data.x.tensor
    ((route_present r.preparation).1 _ (by simp [reads])) i.val (by rw [←r.preparation.data.x.countEq]; exact hi)
  exact hk.trans (ht.trans (prepare_x_real r.preparation m i.val i.isLt))

theorem body_saved_w_real (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (present : initial.live w) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin w.view.shape.prod) :
    Storage.realLogical (r.realFinal m epsReal) r.context.w.tensor i.val=Storage.realLogical m w i.val := by
  have hi : i.val<r.preparation.data.w.count := by change i.val<r.preparation.wr.output.count; rw [(materialize_metadata r.preparation.wr).2.1]; exact i.isLt
  have hk := body_real_input r m epsReal r.preparation.data.w (by simp [reads]) ⟨i.val,hi⟩
  have ht := r.dispatch.preTransition.realValues (r.preparation.transition.shadow m) r.preparation.data.w.tensor
    ((route_present r.preparation).1 _ (by simp [reads])) i.val (by rw [←r.preparation.data.w.countEq]; exact hi)
  exact hk.trans (ht.trans (prepare_w_real r.preparation present m i.val i.isLt))

variable {metadata : MetadataAPI api neg inputs Metadata}

theorem saved_live (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (t : Dense) (member : t∈[r.body.context.x,r.body.context.w,r.body.context.r,r.body.context.mu]++r.body.context.bias.toList) :
    r.final.live t.tensor :=
  (lib.leave r.sShape.owner).transition r.body.final |>.live _ (body_saved_live r.body t member)

theorem saved_real_post (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (t : Dense)
    (member : t∈[r.body.context.x,r.body.context.w,r.body.context.r,r.body.context.mu]++r.body.context.bias.toList)
    (i : Fin t.count) :
    Storage.realLogical (r.realFinal m epsReal) t.tensor i.val=
      Storage.realLogical (r.body.realFinal (r.preTransition.shadow m) epsReal) t.tensor i.val :=
  (lib.leave r.sShape.owner).transition r.body.final |>.realValues _ _ (body_saved_live r.body t member) i.val
    (by rw [←t.countEq]; exact i.isLt)

theorem saved_x_real (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (present : initial.live x) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin x.view.shape.prod) :
    Storage.realLogical (r.realFinal m epsReal) r.body.context.x.tensor i.val=Storage.realLogical m x i.val := by
  have hi : i.val<r.body.context.x.count := by
    change i.val<r.body.preparation.xr.output.count
    rw [(materialize_metadata r.body.preparation.xr).2.1]
    exact i.isLt
  exact (saved_real_post r m epsReal _ (by simp) ⟨i.val,hi⟩).trans
    ((body_saved_x_real r.body _ epsReal i).trans (r.preTransition.realValues m x present i.val i.isLt))

theorem saved_w_real (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (present : initial.live w) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin w.view.shape.prod) :
    Storage.realLogical (r.realFinal m epsReal) r.body.context.w.tensor i.val=Storage.realLogical m w i.val := by
  have hi : i.val<r.body.context.w.count := by
    change i.val<r.body.preparation.wr.output.count
    rw [(materialize_metadata r.body.preparation.wr).2.1]
    exact i.isLt
  exact (saved_real_post r m epsReal _ (by simp) ⟨i.val,hi⟩).trans
    ((body_saved_w_real r.body (r.preTransition.live w present) _ epsReal i).trans
      (r.preTransition.realValues m w present i.val i.isLt))

theorem original_row_transport (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (present : initial.live x) (m : Storage.World (Nat → ℝ)) (row col : Nat)
    (hr : row<r.shape.rows) (hc : col<r.shape.columns) :
    logicalX (r.preTransition.shadow m) x r.shape.columns row col=logicalX m x r.shape.columns row col := by
  have hi : row*r.shape.columns+col<x.view.shape.prod := by
    have hs := shape_count r.shape
    have hm := Nat.mul_le_mul_right r.shape.columns (show row+1≤r.shape.rows by omega)
    nlinarith
  exact r.preTransition.realValues m x present _ hi

theorem saved_mean_real (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (present : initial.live x) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (row : Nat) (hr : row<r.shape.rows) :
    Storage.realLogical (r.realFinal m epsReal) r.body.context.mu.tensor row=
      RealProof.meanSpec r.shape.columns (logicalX m x r.shape.columns row) := by
  have hi : row<r.body.context.mu.count := by rw [(body_saved_layout r.body).muCount]; exact hr
  have h := (saved_real_post r m epsReal _ (by simp) ⟨row,hi⟩).trans
    (body_real_mean r.body (r.preTransition.shadow m) epsReal row hr)
  exact h.trans (RealProof.mean_congr _ _ _ (fun col hc => original_row_transport r present m row col hr hc))

theorem saved_inverse_real (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (present : initial.live x) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (row : Nat) (hr : row<r.shape.rows) :
    Storage.realLogical (r.realFinal m epsReal) r.body.context.r.tensor row=
      RealProof.invSpec r.shape.columns (logicalX m x r.shape.columns row) epsReal := by
  have hi : row<r.body.context.r.count := by rw [(body_saved_layout r.body).rCount]; exact hr
  have h := (saved_real_post r m epsReal _ (by simp) ⟨row,hi⟩).trans
    (body_real_inverse r.body (r.preTransition.shadow m) epsReal row hr)
  exact h.trans (RealProof.inv_congr _ _ _ (fun col hc => original_row_transport r present m row col hr hc) epsReal)

#axiomsClean body_real_input
#axiomsClean saved_x_real
#axiomsClean saved_w_real
#axiomsClean saved_mean_real
#axiomsClean saved_inverse_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
