import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardPreparationValues

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : Service platform api neg inputs Metadata}
  {c : SavedContext} {base : Base} {scratch : Scratch} {layout : BaseLayout base c inputs}
  {sl : ScratchLayout base scratch c inputs} {initial : State Store Metadata}

def LaunchesRoute.firstReal (r : LaunchesRoute service c base scratch layout sl initial) (m : Storage.World (Nat → ℝ)) :=
  (groupBuffers base scratch layout sl).buffers.run Memory.realAPI
    (Execution.groupValues RealProof.ops r.first.code.plan Memory.realAPI (groupBuffers base scratch layout sl)
      c.x c.w base.dy c.r c.mu (r.first.preTransition.shadow m)) r.first.schedule (r.first.preTransition.shadow m)

def LaunchesRoute.realFinal (r : LaunchesRoute service c base scratch layout sl initial) (m : Storage.World (Nat → ℝ)) :=
  let after := r.last.preTransition.shadow (r.firstReal m)
  (finalBuffers base layout).buffers.run Memory.realAPI
    (Execution.finalValues RealProof.ops r.last.code.finalPlan Memory.realAPI (finalBuffers base layout)
      (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db after) r.last.schedule after

theorem transition_dense_real {s t : State Store Metadata} (tr : Transition api neg inputs s t)
    (d : Dense) (hp : s.live d.tensor) (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<d.count) :
    Execution.E.vector d (tr.shadow m) Memory.realAPI i=Execution.E.vector d m Memory.realAPI i := by
  have h := tr.realValues m d.tensor hp i (by rw [←d.countEq]; exact hi)
  simpa only [dense_real _ _ _ hi] using h

theorem launches_scratch_real (r : LaunchesRoute service c base scratch layout sl initial)
    (live : initial.live scratch.dw.tensor ∧ initial.live scratch.db.tensor)
    (m : Storage.World (Nat → ℝ)) (group col : Nat)
    (hg : group<min (Geometry.groups c.columns) c.rows) (hc : col<c.columns) :
    Execution.E.rows scratch.dw c.columns (r.last.preTransition.shadow (r.firstReal m)) Memory.realAPI group col=
      (groupValue RealProof.ops r.first.code.plan c.rows (Geometry.groups c.columns) group
        (Execution.E.rows c.x c.columns (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.rows base.dy c.columns (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.saved Memory.realAPI c.r c.mu (r.first.preTransition.shadow m)) col).1 ∧
    Execution.E.rows scratch.db c.columns (r.last.preTransition.shadow (r.firstReal m)) Memory.realAPI group col=
      (groupValue RealProof.ops r.first.code.plan c.rows (Geometry.groups c.columns) group
        (Execution.E.rows c.x c.columns (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.rows base.dy c.columns (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.saved Memory.realAPI c.r c.mu (r.first.preTransition.shadow m)) col).2 := by
  have hi : group*c.columns+col<min (Geometry.groups c.columns) c.rows*c.columns := by
    have hmul := Nat.mul_le_mul_right c.columns (show group+1≤min (Geometry.groups c.columns) c.rows by omega)
    nlinarith
  have hw := transition_dense_real r.last.preTransition scratch.dw (r.first.preTransition.live _ live.1)
    (r.firstReal m) _ (by simpa only [sl.dwCount] using hi)
  have hb := transition_dense_real r.last.preTransition scratch.db (r.first.preTransition.live _ live.2)
    (r.firstReal m) _ (by simpa only [sl.dbCount] using hi)
  have hr := Execution.group_readback RealProof.ops r.first.code.plan Memory.realAPI
    (groupBuffers base scratch layout sl) c.x c.w base.dy c.r c.mu (r.first.preTransition.shadow m)
    r.first.schedule (Runtime.route_coverage r.first) group col hg hc
  exact ⟨hw.trans hr.1,hb.trans hr.2⟩

theorem launches_dw_real (r : LaunchesRoute service c base scratch layout sl initial)
    (live : initial.live scratch.dw.tensor ∧ initial.live scratch.db.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Fin base.dw.count) :
    Storage.realLogical (r.realFinal m) base.dw.tensor i.val=
      RealProof.Spec.weightGradient c.columns c.rows
        (Execution.E.rows c.x c.columns (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.rows base.dy c.columns (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.saved Memory.realAPI c.r c.mu (r.first.preTransition.shadow m)) i.val := by
  rw [dense_real _ _ _ i.isLt]
  have ho : base.dw.owner=c.x.owner := layout.owners 1
  rw [ho]
  have read := Execution.run_readback (finalBuffers base layout).buffers Memory.realAPI
    (Execution.finalValues RealProof.ops r.last.code.finalPlan Memory.realAPI (finalBuffers base layout)
      (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db (r.last.preTransition.shadow (r.firstReal m)))
    r.last.schedule (Runtime.route_coverage r.last) (r.last.preTransition.shadow (r.firstReal m)) 0 i
  apply read.trans
  have hc : i.val<c.columns := by simpa only [layout.dwCount] using i.isLt
  apply Eq.trans (Execution.final_congr RealProof.ops r.last.code.finalPlan _ _ _
    (fun group hg => (launches_scratch_real r live m group i.val hg hc).1) i.val)
  exact RealProof.weightGradient_real r.first.code.plan r.last.code.finalPlan c.rows (Geometry.groups c.columns)
    (by simp only [Geometry.groups]; split_ifs <;> norm_num) _ _ _ _ hc

theorem launches_db_real (r : LaunchesRoute service c base scratch layout sl initial)
    (live : initial.live scratch.dw.tensor ∧ initial.live scratch.db.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Fin base.db.count) :
    Storage.realLogical (r.realFinal m) base.db.tensor i.val=
      RealProof.Spec.biasGradient c.columns c.rows
        (Execution.E.rows base.dy c.columns (r.first.preTransition.shadow m) Memory.realAPI) i.val := by
  rw [dense_real _ _ _ i.isLt]
  have ho : base.db.owner=c.x.owner := layout.owners 2
  rw [ho]
  have read := Execution.run_readback (finalBuffers base layout).buffers Memory.realAPI
    (Execution.finalValues RealProof.ops r.last.code.finalPlan Memory.realAPI (finalBuffers base layout)
      (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db (r.last.preTransition.shadow (r.firstReal m)))
    r.last.schedule (Runtime.route_coverage r.last) (r.last.preTransition.shadow (r.firstReal m)) 1 i
  apply read.trans
  have hc : i.val<c.columns := by simpa only [layout.dbCount] using i.isLt
  apply Eq.trans (Execution.final_congr RealProof.ops r.last.code.finalPlan _ _ _
    (fun group hg => (launches_scratch_real r live m group i.val hg hc).2) i.val)
  exact RealProof.biasGradient_real r.first.code.plan r.last.code.finalPlan c.rows (Geometry.groups c.columns)
    (by simp only [Geometry.groups]; split_ifs <;> norm_num) _ _ _ _ hc

theorem launches_dx_real (r : LaunchesRoute service c base scratch layout sl initial)
    (live : initial.live base.dx.tensor) (m : Storage.World (Nat → ℝ)) (i : Fin base.dx.count) :
    Storage.realLogical (r.realFinal m) base.dx.tensor i.val=
      RealProof.Spec.gradient c.columns
        (Execution.E.rows c.x c.columns (r.first.preTransition.shadow m) Memory.realAPI (i.val/c.columns))
        (Execution.E.rows base.dy c.columns (r.first.preTransition.shadow m) Memory.realAPI (i.val/c.columns))
        (Execution.E.vector c.w (r.first.preTransition.shadow m) Memory.realAPI)
        (Execution.E.saved Memory.realAPI c.r c.mu (r.first.preTransition.shadow m) (i.val/c.columns)).mean
        (Execution.E.saved Memory.realAPI c.r c.mu (r.first.preTransition.shadow m) (i.val/c.columns)).inverse (i.val%c.columns) := by
  rw [dense_real _ _ _ i.isLt]
  have ho : base.dx.owner=c.x.owner := layout.owners 0
  have fresh : ∀ k,Storage.Disjoint ((finalBuffers base layout).buffers.output k).tensor.backing base.dx.tensor.backing := by
    intro k; fin_cases k
    · exact layout.separated 1 0 (by decide)
    · exact layout.separated 2 0 (by decide)
  have hp := Execution.run_input (finalBuffers base layout).buffers Memory.realAPI
    (Execution.finalValues RealProof.ops r.last.code.finalPlan Memory.realAPI (finalBuffers base layout)
      (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db (r.last.preTransition.shadow (r.firstReal m)))
    r.last.schedule (r.last.preTransition.shadow (r.firstReal m)) base.dx ho fresh i
  apply hp.trans
  apply (transition_dense_real r.last.preTransition base.dx (r.first.preTransition.live _ live) (r.firstReal m) i.val i.isLt).trans
  change r.firstReal m base.dx.owner (base.dx.address i.val)=_
  rw [ho]
  exact (Execution.run_readback (groupBuffers base scratch layout sl).buffers Memory.realAPI
    (Execution.groupValues RealProof.ops r.first.code.plan Memory.realAPI (groupBuffers base scratch layout sl)
      c.x c.w base.dy c.r c.mu (r.first.preTransition.shadow m)) r.first.schedule (Runtime.route_coverage r.first)
      (r.first.preTransition.shadow m) 0 i).trans (RealProof.backward_real r.first.code.plan _ _ _ _ _)

#axiomsClean launches_scratch_real
#axiomsClean launches_dx_real
#axiomsClean launches_dw_real
#axiomsClean launches_db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
