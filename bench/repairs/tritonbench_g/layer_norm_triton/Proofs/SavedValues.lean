import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Saved

namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation Layout
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : ForwardService platform api neg inputs Metadata}
  {p : Preparation.Data} {layout : Prepared p M N inputs}
  {settings : Compiler.settings N=.ok (B,Geometry.warps B)} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

theorem launch_input_real (r : LaunchRoute service p layout settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (t : Dense) (member : t∈Layout.reads p) (i : Fin t.count) :
    Storage.realLogical (r.realFinal m epsilon) t.tensor i.val=Storage.realLogical (r.preReal m) t.tensor i.val := by
  cases r with
  | skipped _ => rfl
  | ran active lr =>
    let l := layout.nonemptyLayout (by omega)
    rw [dense_real _ _ _ i.isLt,dense_real _ _ _ i.isLt]
    exact Execution.run_input (Execution.forwardBuffers p l) Memory.realAPI
      (Execution.forwardValues RealProof.ops lr.code.plan Memory.realAPI p epsilon (lr.preTransition.shadow m))
      lr.schedule (lr.preTransition.shadow m) t (l.inputsOwned t member) (fun k => l.freshInputs k t member) i

theorem launch_stats_real (r : LaunchRoute service p layout settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (hn : 0<N) (row : Nat) (hr : row<M) :
    Execution.E.saved Memory.realAPI p.r p.mu (r.realFinal m epsilon) row=
      ⟨RealProof.Spec.mean N (Execution.E.rows p.x N (r.preReal m) Memory.realAPI row),
       RealProof.Spec.inverse N (Execution.E.rows p.x N (r.preReal m) Memory.realAPI row) epsilon⟩ := by
  cases r with
  | skipped inactive => exact False.elim (inactive ⟨by omega,by omega⟩)
  | ran active lr =>
    exact Execution.forward_saved_real lr.code.plan p (layout.nonemptyLayout hn) epsilon
      (lr.preTransition.shadow m) lr.schedule (Runtime.route_coverage lr) row hr

variable {lib : APIHost api neg inputs Metadata} {identity : Nat} {x w b : Storage.Tensor}
  {shape : Geometry.Shape x.view.shape w.view.shape b.view.shape} {B : Nat}
  {settings : Compiler.settings shape.columns=.ok (B,Geometry.warps B)}

theorem body_input_real (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (t : Dense) (member : t∈Layout.reads r.preparation.data)
    (i : Fin t.count) :
    Storage.realLogical (r.realFinal m epsilon) t.tensor i.val=
      Storage.realLogical (r.launch.preReal (r.preparation.transition.shadow m)) t.tensor i.val :=
  (r.tail.realValues _ _ (launch_live r.launch _ ((Layout.route_present r.preparation).1 t member)) i.val
    (by rw [←t.countEq]; exact i.isLt)).trans (launch_input_real r.launch _ epsilon t member i)

theorem body_stats_real (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (hn : 0<shape.columns) (row : Nat) (hr : row<shape.rows) :
    ⟨Storage.realLogical (r.realFinal m epsilon) r.context.mu.tensor row,
     Storage.realLogical (r.realFinal m epsilon) r.context.r.tensor row⟩ =
      (⟨RealProof.Spec.mean shape.columns (logicalRow m x shape.columns row),
        RealProof.Spec.inverse shape.columns (logicalRow m x shape.columns row) epsilon⟩ : VeriTile.Bench.Repairs.FastLayernorm.Statistics ℝ) := by
  let l := r.layout.nonemptyLayout hn
  have hmu : row<r.preparation.data.mu.count := by simpa only [l.muCount] using hr
  have hinv : row<r.preparation.data.r.count := by simpa only [l.rCount] using hr
  have hm := r.tail.realValues (r.launch.realFinal (r.preparation.transition.shadow m) epsilon) r.preparation.data.mu.tensor
    (launch_live r.launch _ ((Layout.route_present r.preparation).2 1)) row
    (by rw [←r.preparation.data.mu.countEq]; exact hmu)
  have hv := r.tail.realValues (r.launch.realFinal (r.preparation.transition.shadow m) epsilon) r.preparation.data.r.tensor
    (launch_live r.launch _ ((Layout.route_present r.preparation).2 0)) row
    (by rw [←r.preparation.data.r.countEq]; exact hinv)
  have stats := launch_stats_real r.launch (r.preparation.transition.shadow m) epsilon hn row hr
  have hm0 := congrArg VeriTile.Bench.Repairs.FastLayernorm.Statistics.mean stats
  have hv0 := congrArg VeriTile.Bench.Repairs.FastLayernorm.Statistics.inverse stats
  apply congrArg₂ VeriTile.Bench.Repairs.FastLayernorm.Statistics.mk
  · apply hm.trans
    rw [dense_real _ _ _ hmu]
    apply hm0.trans
    exact VeriTile.Bench.Repairs.FastLayernorm.RealProof.mean_congr _ _ _ (fun col hc => source_x_real r m row col hr hc)
  · apply hv.trans
    rw [dense_real _ _ _ hinv]
    apply hv0.trans
    exact VeriTile.Bench.Repairs.FastLayernorm.RealProof.inv_congr _ _ _ (fun col hc => source_x_real r m row col hr hc) epsilon

variable {normalized : List Nat}

theorem body_saved_live (r : Call lib service identity x w b normalized eps initial)
    (t : Dense) (ht : t∈[r.body.context.x,r.body.context.w,r.body.context.r,r.body.context.mu]++r.body.context.bias.toList) :
    r.body.final.live t.tensor := by
  apply r.body.tail.live
  apply launch_live
  simp only [BodyRoute.context,savedContext,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with (hx|hw|hr|hm)|hb
  · subst t; exact (Layout.route_present r.body.preparation).1 _ (by simp [Layout.reads])
  · subst t; exact (Layout.route_present r.body.preparation).1 _ (by simp [Layout.reads])
  · subst t; exact (Layout.route_present r.body.preparation).2 0
  · subst t; exact (Layout.route_present r.body.preparation).2 1
  · exact (Layout.route_present r.body.preparation).1 _ (by simp [Layout.reads,hb])

theorem saved_post_real (r : Call lib service identity x w b normalized eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (t : Dense)
    (ht : t∈[r.body.context.x,r.body.context.w,r.body.context.r,r.body.context.mu]++r.body.context.bias.toList) (i : Fin t.count) :
    Storage.realLogical (r.realFinal m epsilon) t.tensor i.val=
      Storage.realLogical (r.body.realFinal (((lib.enter x.view.device).transition initial).shadow m) epsilon) t.tensor i.val :=
  (lib.leave initial.owner).transition r.body.final |>.realValues _ _ (body_saved_live r t ht) i.val
    (by rw [←t.countEq]; exact i.isLt)

theorem saved_x_real (r : Call lib service identity x w b normalized eps initial) (present : initial.live x)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (row col : Nat) (hr : row<r.shape.rows) (hc : col<r.shape.columns) :
    Storage.realLogical (r.realFinal m epsilon) r.body.context.x.tensor (row*r.shape.columns+col)=
      logicalRow m x r.shape.columns row col := by
  have hi : row*r.shape.columns+col<x.view.shape.prod := by
    rw [(Geometry.shape_product r.shape).1]
    have hmul := Nat.mul_le_mul_right r.shape.columns (show row+1≤r.shape.rows by omega)
    nlinarith
  have hid : row*r.shape.columns+col<r.body.context.x.count := by
    change _<r.body.preparation.xr.output.count
    rw [(materialize_metadata r.body.preparation.xr).2.1]
    exact hi
  let tr := (lib.enter x.view.device).transition initial
  apply (saved_post_real r m epsilon _ (by simp) ⟨_,hid⟩).trans
  apply (body_input_real r.body (tr.shadow m) epsilon _ (by simp [Layout.reads,BodyRoute.context,savedContext]) ⟨_,hid⟩).trans
  rw [dense_real _ _ _ hid]
  exact (source_x_real r.body (tr.shadow m) row col hr hc).trans (tr.realValues m x present _ hi)

theorem saved_w_real (r : Call lib service identity x w b normalized eps initial) (present : initial.live w)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (col : Nat) (hc : col<r.shape.columns) :
    Storage.realLogical (r.realFinal m epsilon) r.body.context.w.tensor col=Storage.realLogical m w col := by
  have hi : col<w.view.shape.prod := by simpa only [(Geometry.shape_product r.shape).2.1] using hc
  have hid : col<r.body.context.w.count := by
    change _<r.body.preparation.wr.output.count
    rw [(materialize_metadata r.body.preparation.wr).2.1]
    exact hi
  let tr := (lib.enter x.view.device).transition initial
  apply (saved_post_real r m epsilon _ (by simp) ⟨_,hid⟩).trans
  apply (body_input_real r.body (tr.shadow m) epsilon _ (by simp [Layout.reads,BodyRoute.context,savedContext]) ⟨_,hid⟩).trans
  rw [dense_real _ _ _ hid]
  exact (source_w_real r.body (tr.live _ present) (tr.shadow m) col hc).trans (tr.realValues m w present _ hi)

theorem saved_stats_real (r : Call lib service identity x w b normalized eps initial) (present : initial.live x)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (hn : 0<r.shape.columns) (row : Nat) (hr : row<r.shape.rows) :
    ⟨Storage.realLogical (r.realFinal m epsilon) r.body.context.mu.tensor row,
     Storage.realLogical (r.realFinal m epsilon) r.body.context.r.tensor row⟩ =
      (⟨RealProof.Spec.mean r.shape.columns (logicalRow m x r.shape.columns row),
        RealProof.Spec.inverse r.shape.columns (logicalRow m x r.shape.columns row) epsilon⟩ : VeriTile.Bench.Repairs.FastLayernorm.Statistics ℝ) := by
  let l := r.body.layout.nonemptyLayout hn
  have hmu : row<r.body.context.mu.count := by simpa only [BodyRoute.context,savedContext,l.muCount] using hr
  have hinv : row<r.body.context.r.count := by simpa only [BodyRoute.context,savedContext,l.rCount] using hr
  let tr := (lib.enter x.view.device).transition initial
  have hs := body_stats_real r.body (tr.shadow m) epsilon hn row hr
  have hx : ∀ col,col<r.shape.columns → logicalRow (tr.shadow m) x r.shape.columns row col=logicalRow m x r.shape.columns row col := by
    intro col hc
    apply tr.realValues m x present
    rw [(Geometry.shape_product r.shape).1]
    have hmul := Nat.mul_le_mul_right r.shape.columns (show row+1≤r.shape.rows by omega)
    nlinarith
  apply congrArg₂ VeriTile.Bench.Repairs.FastLayernorm.Statistics.mk
  · apply (saved_post_real r m epsilon _ (by simp) ⟨_,hmu⟩).trans
    exact (congrArg VeriTile.Bench.Repairs.FastLayernorm.Statistics.mean hs).trans
      (VeriTile.Bench.Repairs.FastLayernorm.RealProof.mean_congr _ _ _ hx)
  · apply (saved_post_real r m epsilon _ (by simp) ⟨_,hinv⟩).trans
    exact (congrArg VeriTile.Bench.Repairs.FastLayernorm.Statistics.inverse hs).trans
      (VeriTile.Bench.Repairs.FastLayernorm.RealProof.inv_congr _ _ _ hx epsilon)

#axiomsClean saved_x_real
#axiomsClean saved_w_real
#axiomsClean saved_stats_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
