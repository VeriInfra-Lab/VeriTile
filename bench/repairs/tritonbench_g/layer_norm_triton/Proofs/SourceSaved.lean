import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardPublic
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelBackward

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceSaved
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section

theorem saved [Inhabited α] (model : SourceModel.Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats api neg r mu (SourceForwardStorage.final model B eps api neg c memory) row=
      ⟨SourceForwardGraph.mean model N B (KernelForward.row api neg c.x memory N row),
       SourceForwardGraph.inverse model N B eps (KernelForward.row api neg c.x memory N row)⟩ := by
  have ri : row<r.dense.count := by rw [rBinding,c.inverseCount]; exact hr
  have mi : row<mu.dense.count := by rw [muBinding,c.meanCount]; exact hr
  have ir : row<(c.buffers.output 0).count := by rw [c.inverseCount]; exact hr
  have im : row<(c.buffers.output 1).count := by rw [c.meanCount]; exact hr
  have rv := KernelMemory.run_read c.buffers api neg
    (SourceForwardStorage.graphValues model B eps api neg c memory) memory 0 ⟨row,ir⟩
  have mv := KernelMemory.run_read c.buffers api neg
    (SourceForwardStorage.graphValues model B eps api neg c memory) memory 1 ⟨row,im⟩
  unfold KernelBackward.initialStats
  rw [inputValue_logical api neg mu _ row mi,inputValue_logical api neg r _ row ri]
  simp only [rBinding,muBinding,SourceForwardStorage.final,launch,if_true]
  rw [rv,mv]
  rfl

theorem saved_real (model : SourceModel.Model ℝ) (laws : SourceForwardReal.Laws model) (B : Nat) (eps : ℝ)
    (c : KernelContract.Forward M N) (memory : World (Nat → ℝ)) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (hb : N≤B) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats Memory.realAPI Neg.neg r mu
      (SourceForwardStorage.final model B eps Memory.realAPI Neg.neg c memory) row=
      ⟨RealProof.Spec.mean N (KernelForward.row Memory.realAPI Neg.neg c.x memory N row),
       RealProof.Spec.inverse N (KernelForward.row Memory.realAPI Neg.neg c.x memory N row) eps⟩ := by
  rw [saved model B eps Memory.realAPI Neg.neg c memory r mu rBinding muBinding launch row hr,
    SourceForwardReal.mean_real model laws N B hb,SourceForwardReal.inverse_real model laws N B hb]

#axiomsClean saved
#axiomsClean saved_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceSaved
