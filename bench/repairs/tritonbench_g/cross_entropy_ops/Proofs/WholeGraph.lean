import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ForwardGraph
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.GradientGraph

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton Source
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Observations TensorArithmetic TensorLibrary CollectiveLibrary IntegerViews
  ObjectRegistry FloatArray IntArray SavedPreserves value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Complete actual source/host expression graph for both forward outputs and
backward. The saved global LSE is derived from this same forward; every load,
scale product and host barrier survives in arbitrary arithmetic. -/
theorem complete_graph {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : TensorArithmetic α)
    (rt : Runtime α σ) (lib : LocalAPI obs rt) (ops : TensorLibrary obs A rt.toRuntime)
    (zeros : ZeroAPI obs rt A.zero) (ints : IntegerViews obs) (objects : ObjectRegistry rt.toRuntime)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial saved : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned saved} (k : ForwardKernelData obs f)
    (views : ForwardViewData obs objects f) (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (size : f.world>1 → f.world=gs (o.processGroup.getD 0)) (aggregation : ForwardAggregation f)
    (lossPeers lsePeers : Nat → α) (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    {before after : World α σ} (between : SavedPreserves obs returned.context.base saved before)
    (callerUpstream output : Tensor) (upstreamLive : before.live callerUpstream.region)
    (upstreamShape : callerUpstream.shape=[f.rows]) (upstreamDtype : callerUpstream.dtype="torch.float32")
    (b : BackwardData M rt returned.context callerUpstream output before after) :
    (∀ row, row<f.rows → value obs saved returned.losses row=finalLoss k A lossPeers lsePeers row ∧
      value obs saved returned.zLosses row=finalZ k A lsePeers row) ∧
    (∀ row, row<f.rows → ∀ col, col<f.cols → value obs after output (row*f.cols+col)=
      gradientScalar M b.parameters (obs.integer initial labels row) (value obs initial logits (row*f.cols+col))
        (finalLSE k A lsePeers row) (value obs before callerUpstream row) col) := by
  have forward := forward_graph_values obs A rt lib ops zeros ints objects gs gr comms k views labelDtype size aggregation lossPeers lsePeers inputs
  have savedData := forward.2.2.2
  have ctx := between.context savedData.context
  obtain ⟨bk⟩ := backward_kernel_data obs rt lib objects b ctx upstreamLive
  have forwardShape : returned.context.logits.shape=[f.rows,f.cols] := savedData.logits.metadata.shape.trans f.logitsShape
  have shapeEq : [b.rows,b.cols]=[f.rows,f.cols] := b.shape.symm.trans forwardShape
  have dimensions : b.rows=f.rows ∧ b.cols=f.cols := by simpa only [List.cons.injEq,and_true] using shapeEq
  have savedLSE : FloatArray obs before returned.context.lse [b.rows] (finalLSE k A lsePeers) := by
    simpa only [dimensions.1] using between.lse forward.2.1
  have savedLabels : IntArray obs before returned.context.labels [b.rows] (obs.integer initial labels) := by
    simpa only [dimensions.1] using between.labels savedData.labels
  have backward := backward_graph_values obs rt lib ints bk ctx savedData.layout
    (savedData.logits.metadata.dtype.trans dtype) savedData.lseDense savedData.labelsDense
    (obs.integer initial labels) (finalLSE k A lsePeers) savedLSE savedLabels
    (by simpa only [dimensions.1] using upstreamShape) upstreamDtype
  constructor
  · intro row hr
    exact ⟨forward.1.values row (by simpa [Tensor.numel,forward.1.shapeEq] using hr),
      forward.2.2.1.values row (by simpa [Tensor.numel,forward.2.2.1.shapeEq] using hr)⟩
  · intro row hr col hc
    have result := backward row (by simpa only [dimensions.1] using hr) col (by simpa only [dimensions.2] using hc)
    simp only [dimensions.2] at result
    rw [result]
    have bound : row*f.cols+col<logits.numel := by
      simp only [Tensor.numel,f.logitsShape,List.prod_cons,List.prod_nil,Nat.mul_one]
      have product := Nat.mul_le_mul_right f.cols (Nat.succ_le_of_lt hr)
      simp only [Nat.succ_mul] at product
      omega
    rw [between.floatValue returned.context.logits (Or.inl rfl),savedData.logits.floats dtype _ bound]

#axiomsClean complete_graph
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
