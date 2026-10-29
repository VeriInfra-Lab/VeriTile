import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardReads
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy1.Source
open VeriTile.Triton
noncomputable section

/-- One visible gradient's source operation graph. Loads remain explicit,
including the saved forward LSE and actual upstream snapshot. -/
def gradientScalar {α : Type} (M : FP32 α) (p : Parameters) (target : Int)
    (x lse upstream : α) (col : Nat) : α :=
  if target=p.ignored then fp M 0 else
  let probability := unary M .exp (binary M .sub (M.load x) (M.load lse))
  let corrected := binary M .add probability
    (binary M .mul (binary M .mul (binary M .mul (fp M 2) (fp M p.squareScale)) (M.load lse)) probability)
  let adjusted := if p.hasSmoothing then
      binary M .sub
        (if (col : Int)=target-p.classStart then
          binary M .sub corrected (binary M .sub (nat M 1) (fp M p.smoothing)) else corrected)
        (binary M .div (fp M p.smoothing) (nat M p.totalClasses))
    else if (col : Int)=target-p.classStart then binary M .sub corrected (fp M 1) else corrected
  binary M .mul (M.load upstream) adjusted

theorem gradient_scalar_source {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (memory : Memory α)
    (row block : Nat) (lane : Fin p.block) (active : column p block lane.val<p.cols)
    (target : Int) (x lse upstream : α)
    (labelRead : label p memory row=target)
    (xRead : (memory p.logits (logitAddress p row (column p block lane.val))).read .real=x)
    (lseRead : (memory p.lse (p.lseBase+row)).read .real=lse)
    (upstreamRead : (memory p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=upstream) :
    backwardValues M p memory row block lane=gradientScalar M p target x lse upstream (column p block lane.val) := by
  by_cases ignored : target=p.ignored
  · simp [backwardValues,gradientScalar,labelRead,ignored]
  · simp only [backwardValues,gradientScalar,labelRead,if_neg ignored,loadedLogits,if_pos active,load,xRead,lseRead,upstreamRead]

#axiomsClean gradient_scalar_source
end
end VeriTile.Bench.Repairs.CrossEntropy1.Source

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem backward_graph_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (ints : IntegerViews obs) {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects b)
    (ctx : BackwardContext rt objects c initial) (layout : RowLayout c.logits) (dtype : c.logits.dtype="torch.float32")
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
  have coordinates := RealHost.block_coordinates b.cols col row hc
  simp only [BackwardKernelData.gradientValues,coordinates.1,coordinates.2]
  have positive : 0<b.parameters.block := Nat.pos_of_ne_zero b.nonzero
  let block := col/b.parameters.block
  let lane : Fin b.parameters.block := ⟨col%b.parameters.block,Nat.mod_lt col positive⟩
  have columnEq : column b.parameters block lane.val=col := by
    simpa only [column,block,lane,Nat.mul_comm b.parameters.block] using Nat.div_add_mod col b.parameters.block
  have lseBound : row<c.lse.numel := by simpa [Tensor.numel,savedLSE.shapeEq] using hr
  have lseRead := (backward_lse_read obs k
    ⟨savedLSE.shapeEq,savedLSE.dtype,savedLSE.live,fun _ _ => rfl⟩ lseDense row hr).trans (savedLSE.values row lseBound)
  have source := gradient_scalar_source M b.parameters k.launch.ready.memory row block lane
    (by simpa only [columnEq] using hc) (targets row) (value obs initial c.logits (row*b.cols+col))
    (lses row) (value obs initial callerUpstream row)
    (backward_label_read obs ints k targets savedLabels labelsDense row hr)
    (by simpa only [columnEq] using backward_logit_read obs k ctx layout row col hr hc)
    lseRead (backward_upstream_read obs k upstreamShape upstreamDtype row hr)
  simpa only [columnEq] using source

#axiomsClean backward_graph_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
