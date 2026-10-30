import bench.tritonbench_g.cross_entropy1.Proofs.WholeGraph
import bench.tritonbench_g.cross_entropy1.Proofs.ReturnBacking
import bench.tritonbench_g.cross_entropy1.Proofs.NegativeStorage
import bench.tritonbench_g.cross_entropy1.Proofs.Lowering

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem output_dtype {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {before after : World α σ}
    {b : BackwardData M rt c callerUpstream output before after} (k : BackwardKernelData obs objects b)
    (ctx : BackwardContext rt objects c before) (logitsDtype : c.logits.dtype="torch.float32")
    (originalDtype : c.inplaceLogits.dtype="torch.float32") : output.dtype="torch.float32" := by
  have identity := b.outputIdentity
  by_cases inplace : c.options.inplace=Bool.true
  · simp only [if_pos inplace] at identity
    simpa only [identity] using originalDtype
  · simp only [if_neg inplace] at identity
    have call : rt.emptyLike c.logits Bool.false b.upstreamReady=.ok b.first b.allocated := by
      simpa only [if_neg inplace] using b.firstCall
    have allocated := lib.emptyLike _ _ _ _ _ (k.snapshot.materialized.frame.live _ ctx.logitsLive) call
    simpa only [identity] using allocated.1.dtypeEq.trans logitsDtype

/-- Complete returned byte values for the source and host FP graph. Native
instruction ports are local; the structural lowering theorem supplies the
explicit FMA graph. The saved LSE and all returned-object backing are derived
from preceding actual calls inside this theorem. -/
theorem complete_returned_bytes {σ : Type} (layout : Layout σ) (profile : Floating.ReductionProfile) (n : Floating.Numerics)
    (kept : RegionName → Prop) (obs : Observations Floating.Expr σ)
    (negation : EncodesNegation (Floating.word profile n) obs) (rt : Runtime Floating.Expr σ)
    (lib : Library obs rt) (ops : TensorLibrary obs Floating.hostArithmetic rt) (ints : IntegerViews obs)
    (objects : ObjectRegistry rt) (life : LifetimeLibrary layout rt kept) (physical : BackingLibrary layout rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs Floating.hostArithmetic rt gs gr)
    {o : Options} {logits labels : Tensor} {initial saved : World Floating.Expr σ} {returned : ForwardReturn}
    {f : ForwardData Floating.algebra rt o logits labels initial returned saved} (k : ForwardKernelData obs f)
    (views : ForwardViewData obs objects f) (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (size : f.world>1 → f.world=gs (o.processGroup.getD 0)) (aggregation : ForwardAggregation f)
    (lossPeers lsePeers : Nat → Floating.Expr) (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    {before after : World Floating.Expr σ} (between : SavedPreserves obs returned.context saved before)
    (betweenNative : Retains layout kept saved.native before.native)
    (callerUpstream output : Tensor) (upstreamLive : before.live callerUpstream.region)
    (upstreamShape : callerUpstream.shape=[f.rows]) (upstreamDtype : callerUpstream.dtype="torch.float32")
    (b : BackwardData Floating.algebra rt returned.context callerUpstream output before after)
    (heldLoss : kept returned.losses.region) (heldLogits : kept returned.context.logits.region) (heldWork : kept b.work.region) :
    (∀ row, row<f.rows → ∀ byte : Fin 4,
      bytes layout (Floating.word profile n) saved (layout.buffer saved.native returned.losses.region).device
        ((layout.buffer saved.native returned.losses.region).base+4*returned.losses.index row+byte.val)=
      byteOf (if returned.losses.isNeg then
        flipSign (Floating.runInstruction n (Floating.compile profile (finalLoss k Floating.hostArithmetic lossPeers lsePeers row)).value)
        else Floating.runInstruction n (Floating.compile profile (finalLoss k Floating.hostArithmetic lossPeers lsePeers row)).value) byte) ∧
    (∀ row, row<f.rows → ∀ col, col<f.cols → ∀ byte : Fin 4,
      bytes layout (Floating.word profile n) after (layout.buffer after.native output.region).device
        ((layout.buffer after.native output.region).base+4*output.index (row*f.cols+col)+byte.val)=
      byteOf (if output.isNeg then flipSign (Floating.runInstruction n (Floating.compile profile
        (gradientScalar Floating.algebra b.parameters (obs.integer initial labels row)
          (value obs initial logits (row*f.cols+col)) (finalLSE k Floating.hostArithmetic lsePeers row)
          (value obs before callerUpstream row) col)).value)
        else Floating.runInstruction n (Floating.compile profile
        (gradientScalar Floating.algebra b.parameters (obs.integer initial labels row)
          (value obs initial logits (row*f.cols+col)) (finalLSE k Floating.hostArithmetic lsePeers row)
          (value obs before callerUpstream row) col)).value) byte) := by
  have forward := forward_values obs Floating.hostArithmetic rt lib ops ints gs gr comms k labelDtype size aggregation lossPeers lsePeers inputs
  have graph := complete_graph obs Floating.hostArithmetic rt lib ops ints objects gs gr comms k views dtype labelDtype
    size aggregation lossPeers lsePeers inputs between callerUpstream output upstreamLive upstreamShape upstreamDtype b
  have ctxSaved := forward_backward_context obs Floating.hostArithmetic rt lib ops ints objects gs gr comms k views
    labelDtype size aggregation lossPeers lsePeers inputs
  have ctx := between.context ctxSaved
  obtain ⟨bk⟩ := backward_kernel_data obs rt lib objects b ctx upstreamLive
  have logitsDtype := forward.2.2.2.1.metadata.dtype.trans dtype
  have originalEq : returned.context.inplaceLogits=f.inplaceLogits :=
    congrArg (fun r : ForwardReturn => r.context.inplaceLogits) f.returnIdentity
  have originalDtype : returned.context.inplaceLogits.dtype="torch.float32" := by
    simpa only [originalEq] using views.inplace.metadata.dtype.trans dtype
  have forwardBacking := forward_return_backing layout rt kept life physical k heldLoss forward.1.dtype
  have savedBacking := forward_saved_backing layout rt physical f logitsDtype forward.2.1.dtype forward.2.2.1.dtype originalDtype
  have logitsBacking := savedBacking.1.retain betweenNative heldLogits
  have outputDtype := output_dtype obs rt lib bk ctx logitsDtype originalDtype
  have backwardBacking := backward_return_backing layout rt kept life physical bk heldLogits heldWork logitsDtype outputDtype logitsBacking
  constructor
  · intro row hr byte
    have bound : row<returned.losses.numel := by simpa [Tensor.numel,forward.1.shapeEq] using hr
    have valueBytes := logical_value_bytes layout (Floating.word profile n) obs negation saved returned.losses row byte
      (finalLoss k Floating.hostArithmetic lossPeers lsePeers row) forwardBacking.active (forwardBacking.coordinates row bound) (graph.1 row hr)
    simpa only [Floating.compiled_word] using valueBytes
  · intro row hr col hc byte
    have shape : output.shape=[f.rows,f.cols] := bk.outputShape.trans (forward.2.2.2.1.metadata.shape.trans f.logitsShape)
    have bound := row_coordinate_bound output f.rows f.cols row col shape hr hc
    have valueBytes := logical_value_bytes layout (Floating.word profile n) obs negation after output (row*f.cols+col) byte
      (gradientScalar Floating.algebra b.parameters (obs.integer initial labels row)
        (value obs initial logits (row*f.cols+col)) (finalLSE k Floating.hostArithmetic lsePeers row)
        (value obs before callerUpstream row) col)
      backwardBacking.active (backwardBacking.coordinates _ bound) (graph.2 row hr col hc)
    simpa only [Floating.compiled_word] using valueBytes

#axiomsClean output_dtype
#axiomsClean complete_returned_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
