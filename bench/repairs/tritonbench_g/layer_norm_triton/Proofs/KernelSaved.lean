import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelBackward

/-! The saved-statistics load is bound to the actual forward output Tensor,
with no saved-value equation as an input premise. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.KernelSaved
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section

theorem saved (ops : Ops α) (p : KernelGraph.ForwardPlan N B) (api : Memory.API α Store Byte)
    (neg : α → α) (c : KernelContract.Forward M N) (eps : α) (m : World Store)
    (r mu : Input) (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats api neg r mu (KernelForward.final ops p api neg c eps m) row=
      KernelGraph.stats ops p (KernelForward.row api neg c.x m N row) eps 0 := by
  have ri : row<r.dense.count := by rw [rBinding,c.inverseCount]; exact hr
  have mi : row<mu.dense.count := by rw [muBinding,c.meanCount]; exact hr
  have ir : row<(c.buffers.output 0).count := by rw [c.inverseCount]; exact hr
  have im : row<(c.buffers.output 1).count := by rw [c.meanCount]; exact hr
  have rv := KernelMemory.run_read c.buffers api neg (KernelForward.values ops p api neg c eps m) m 0 ⟨row,ir⟩
  have mv := KernelMemory.run_read c.buffers api neg (KernelForward.values ops p api neg c eps m) m 1 ⟨row,im⟩
  unfold KernelBackward.initialStats
  rw [inputValue_logical api neg mu _ row mi,inputValue_logical api neg r _ row ri]
  simp only [rBinding,muBinding,KernelForward.final,launch,if_true]
  rw [rv,mv]
  rfl

theorem saved_real (p : KernelGraph.ForwardPlan N B) (c : KernelContract.Forward M N)
    (eps : ℝ) (m : World (Nat → ℝ)) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats Memory.realAPI Neg.neg r mu
      (KernelForward.final RealProof.ops p Memory.realAPI Neg.neg c eps m) row=
      ⟨RealProof.Spec.mean N (KernelForward.row Memory.realAPI Neg.neg c.x m N row),
       RealProof.Spec.inverse N (KernelForward.row Memory.realAPI Neg.neg c.x m N row) eps⟩ :=
  (saved RealProof.ops p Memory.realAPI Neg.neg c eps m r mu rBinding muBinding launch row hr).trans
    (KernelGraph.stats_real p _ eps 0)

#axiomsClean saved
#axiomsClean saved_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.KernelSaved
