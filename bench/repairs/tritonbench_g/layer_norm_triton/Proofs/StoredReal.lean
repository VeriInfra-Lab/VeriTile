import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Composition

namespace VeriTile.Bench.Repairs.LayerNormTriton.Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open Storage
noncomputable section

theorem forward_stored_real (plan : Plan N B) (p : Preparation.Data)
    (layout : Layout.Forward p M N inputs) (eps : ℝ) (m : Storage.World (Nat → ℝ))
    (schedule : E.Schedule 3) (covered : ∀ k i,i<(Layout.outputs p k).count → (k,i)∈schedule.order)
    (i : Fin p.y.count) :
    (forwardBuffers p layout).run Memory.realAPI (forwardValues RealProof.ops plan Memory.realAPI p eps m)
      schedule m p.x.owner (p.y.address i.val)=
      RealProof.Spec.output N (E.rows p.x N m Memory.realAPI (i.val/N)) (E.vector p.w m Memory.realAPI)
        (E.vector (p.bias.getD p.w) m Memory.realAPI) eps (i.val%N) := by
  have h := run_readback (forwardBuffers p layout) Memory.realAPI
    (forwardValues RealProof.ops plan Memory.realAPI p eps m) schedule covered m 2 i
  exact h.trans (RealProof.forward_real plan _ _ _ eps _)

theorem forward_saved_real (plan : Plan N B) (p : Preparation.Data)
    (layout : Layout.Forward p M N inputs) (eps : ℝ) (m : Storage.World (Nat → ℝ))
    (schedule : E.Schedule 3) (covered : ∀ k i,i<(Layout.outputs p k).count → (k,i)∈schedule.order)
    (row : Nat) (hr : row<M) :
    E.saved Memory.realAPI p.r p.mu
      ((forwardBuffers p layout).run Memory.realAPI (forwardValues RealProof.ops plan Memory.realAPI p eps m) schedule m) row=
      ⟨RealProof.Spec.mean N (E.rows p.x N m Memory.realAPI row),
       RealProof.Spec.inverse N (E.rows p.x N m Memory.realAPI row) eps⟩ := by
  have hm : row<p.mu.count := by simpa [layout.muCount] using hr
  have hi : row<p.r.count := by simpa [layout.rCount] using hr
  have hmu := run_readback (forwardBuffers p layout) Memory.realAPI
    (forwardValues RealProof.ops plan Memory.realAPI p eps m) schedule covered m 1 ⟨row,hm⟩
  have hinv := run_readback (forwardBuffers p layout) Memory.realAPI
    (forwardValues RealProof.ops plan Memory.realAPI p eps m) schedule covered m 0 ⟨row,hi⟩
  have ho0 : p.r.owner=p.x.owner := layout.outputsOwned 0
  have ho1 : p.mu.owner=p.x.owner := layout.outputsOwned 1
  simp only [forwardValues,RealProof.statistics_real] at hmu hinv
  change VeriTile.Bench.Repairs.FastLayernorm.Statistics.mk _ _ = _
  congr 1
  · simpa only [E.vector,VeriTile.Bench.Repairs.FastLayernorm.Execution.vector,ho1] using hmu
  · simpa only [E.vector,VeriTile.Bench.Repairs.FastLayernorm.Execution.vector,ho0] using hinv

theorem twoLaunch_dx_real (plan : Plan N B) (last : FinalPlan)
    (p : BackwardBuffers M N G inputs) (q : FinalBuffers N inputs)
    (same : (p.buffers.output 0).owner=q.buffers.owner)
    (apart : ∀ k,Storage.Disjoint (q.buffers.output k).tensor.backing (p.buffers.output 0).tensor.backing)
    (x w dy r mu : Dense) (m : Storage.World (Nat → ℝ)) (first : E.Schedule 3) (second : E.Schedule 2)
    (hfirst : ∀ k i,i<(p.buffers.output k).count → (k,i)∈first.order)
    (i : Fin (p.buffers.output 0).count) :
    twoLaunch RealProof.ops plan last Memory.realAPI p q x w dy r mu m first second
      (p.buffers.output 0).owner ((p.buffers.output 0).address i.val)=
      RealProof.Spec.gradient N (E.rows x N m Memory.realAPI (i.val/N))
        (E.rows dy N m Memory.realAPI (i.val/N)) (E.vector w m Memory.realAPI)
        (E.saved Memory.realAPI r mu m (i.val/N)).mean (E.saved Memory.realAPI r mu m (i.val/N)).inverse (i.val%N) := by
  let after := p.buffers.run Memory.realAPI (groupValues RealProof.ops plan Memory.realAPI p x w dy r mu m) first m
  have preserved := run_input q.buffers Memory.realAPI
    (finalValues RealProof.ops last Memory.realAPI q (min G M) (p.buffers.output 1) (p.buffers.output 2) after)
    second after (p.buffers.output 0) same apart i
  apply preserved.trans
  rw [p.buffers.owned 0]
  have h := run_readback p.buffers Memory.realAPI (groupValues RealProof.ops plan Memory.realAPI p x w dy r mu m)
    first hfirst m 0 i
  exact h.trans (RealProof.backward_real plan _ _ _ _ _)

#axiomsClean forward_stored_real
#axiomsClean forward_saved_real
#axiomsClean twoLaunch_dx_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Execution
