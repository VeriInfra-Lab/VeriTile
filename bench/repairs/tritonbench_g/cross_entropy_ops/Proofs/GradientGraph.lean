import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.BackwardReads
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.GradientGraph

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Source
open CrossEntropy1.Source (FP32 Memory)
noncomputable section

/-- Exact scalar source graph, with storage loads before both scale products.
The saved-LSE load remains a barrier; ignored rows select exact zero. -/
def gradientScalar {α : Type} (M : FP32 α) (p : Parameters) (target : Int)
    (x lse upstream : α) (col : Nat) : α :=
  CrossEntropy1.Source.gradientScalar (arithmetic M) p.toParameters target
    (M.mul (M.load x) (M.literal p.logitScale)) (M.load lse)
    (M.mul (M.load upstream) (M.literal p.logitScale)) col

theorem gradient_scalar_source {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (memory : Memory α)
    (row block : Nat) (lane : Fin p.block) (active : CrossEntropy1.Source.column p.toParameters block lane.val<p.cols)
    (target : Int) (x lse upstream : α)
    (labelRead : CrossEntropy1.Source.label p.toParameters memory row=target)
    (xRead : (memory p.logits (CrossEntropy1.Source.logitAddress p.toParameters row (CrossEntropy1.Source.column p.toParameters block lane.val))).read .real=x)
    (lseRead : (memory p.lse (p.lseBase+row)).read .real=lse)
    (upstreamRead : (memory p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=upstream)
    (lseLogits : p.lse≠p.logits) (lseUpstream : p.lse≠p.upstream) :
    backwardValues M p memory row block lane=gradientScalar M p target x lse upstream
      (CrossEntropy1.Source.column p.toParameters block lane.val) := by
  simp only [CrossEntropy1.Source.Cell.read] at xRead lseRead upstreamRead
  apply CrossEntropy1.Source.gradient_scalar_source (arithmetic M) p.toParameters (scaledReads M p memory Bool.true)
    row block lane active target _ _ _ labelRead
  · simp [scaledReads,CrossEntropy1.Source.Cell.read,xRead]
  · simp [scaledReads,CrossEntropy1.Source.Cell.read,lseLogits,lseUpstream,lseRead]
  · simp [scaledReads,CrossEntropy1.Source.Cell.read,upstreamRead]

#axiomsClean gradient_scalar_source
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Source

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton Source
open CrossEntropy1.Source (FP32 column)
open CrossEntropy1.Host (World Observations IntegerViews ObjectRegistry BackwardContext RowLayout FloatArray IntArray value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem backward_graph_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (ints : IntegerViews obs) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects b)
    (ctx : BackwardContext rt.toRuntime objects c.base initial) (layout : RowLayout c.logits) (dtype : c.logits.dtype="torch.float32")
    (lseDense : c.lse.Dense) (labelsDense : c.labels.Dense) (targets : Nat → Int) (lses : Nat → α)
    (savedLSE : FloatArray obs initial c.lse [b.rows] lses) (savedLabels : IntArray obs initial c.labels [b.rows] targets)
    (upstreamShape : callerUpstream.shape=[b.rows]) (upstreamDtype : callerUpstream.dtype="torch.float32") :
    ∀ row, row<b.rows → ∀ col, col<b.cols → value obs final output (row*b.cols+col)=
      gradientScalar M b.parameters (targets row) (value obs initial c.logits (row*b.cols+col))
        (lses row) (value obs initial callerUpstream row) col := by
  have returned := backward_return_values obs rt lib k dtype
  intro row hr col hc
  have bound : row*b.cols+col<output.numel := by
    simp only [Tensor.numel,k.outputShape,b.shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    have product := Nat.mul_le_mul_right b.cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at product
    omega
  rw [returned.2 _ bound]
  have coordinates := CrossEntropy1.RealHost.block_coordinates b.cols col row hc
  simp only [BackwardKernelData.gradientValues,coordinates.1,coordinates.2]
  have positive : 0<b.parameters.block := Nat.pos_of_ne_zero b.nonzero
  let block := col/b.parameters.block
  let lane : Fin b.parameters.block := ⟨col%b.parameters.block,Nat.mod_lt col positive⟩
  have columnEq : column b.parameters.toParameters block lane.val=col := by
    simpa only [column,block,lane,Nat.mul_comm b.parameters.block] using Nat.div_add_mod col b.parameters.block
  have lseBound : row<c.lse.numel := by simpa [Tensor.numel,savedLSE.shapeEq] using hr
  have lseRead := (backward_lse_read obs k
    ⟨savedLSE.shapeEq,savedLSE.dtype,savedLSE.live,fun _ _ => rfl⟩ lseDense row hr).trans (savedLSE.values row lseBound)
  have lseUpstream : b.parameters.lse≠b.parameters.upstream := by
    change c.lse.region≠b.upstream.region
    intro same
    exact k.snapshot.fresh (same ▸ ctx.lseLive)
  have source := gradient_scalar_source M b.parameters k.launch.ready.memory row block lane
    (by simpa only [columnEq] using hc) (targets row) (value obs initial c.logits (row*b.cols+col))
    (lses row) (value obs initial callerUpstream row)
    (backward_label_read obs ints k targets savedLabels labelsDense row hr)
    (by simpa only [columnEq] using backward_logit_read obs k ctx layout row col hr hc)
    lseRead (backward_upstream_read obs k upstreamShape upstreamDtype row hr) ctx.logitsLSE.symm lseUpstream
  simpa only [columnEq] using source

#axiomsClean backward_graph_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
