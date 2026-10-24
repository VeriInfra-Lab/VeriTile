import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SavedUnsplit
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReturnedBackward
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SavedBoundary

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Host (World Step Observations Groups IntegerViews ObjectRegistry ViewGeometry FloatArray value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Complete actual unsplit forward/autograd-backward composition. Saved LSE,
labels, view geometry and separation are derived from this forward's calls.
Only ordinary runtime contracts and caller logical inputs occur as premises. -/
theorem unsplit_forward_backward {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : LocalAPI obs rt) (groups : Groups obs rt.toRuntime)
    (ints : IntegerViews obs) (objects : ObjectRegistry rt.toRuntime) (views : ViewGeometry rt.toRuntime)
    (versions : σ → Nat → Nat) (o : Options) (logits labels upstream output : Tensor)
    (zUpstream : Option Tensor) (initial forwarded backwardInitial final : World (WithBot ℝ) σ) (returned : ForwardReturn)
    (between : CrossEntropy1.Host.SavedPreserves obs returned.context.base forwarded backwardInitial)
    (rows cols : Nat) (shape : logits.shape=[rows,cols])
    (single : ∀ g, o.processGroup=some g → groups.size g=1) (small : cols≤65536)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (strides : logits.strides.length=logits.shape.length) (registered : objects.valid logits)
    (upstreamLive : backwardInitial.live upstream.region) (upstreamShape : upstream.shape=[rows])
    (upstreamDtype : upstream.dtype="torch.float32") (smooth : 0≤o.smoothing)
    (x : Nat → Nat → ℝ) (g : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial logits (row*cols+col)=↑(x row col))
    (targets : ∀ row, row<rows → obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(cols : Int)))
    (upstreams : ∀ row, row<rows → value obs backwardInitial upstream row=↑(g row))
    (forwardCall : forward RealSource.algebra rt o logits labels initial=.ok returned forwarded)
    (backwardCall : backwardEntry RealSource.algebra rt versions returned.context forwarded.native upstream
      zUpstream backwardInitial=.ok output final) :
    returned.losses.shape=[rows] ∧ returned.zLosses.shape=[rows] ∧ output.shape=[rows,cols] ∧
    (o.inplace=Bool.true → output=returned.context.inplaceLogits) ∧
    ∀ row, row<rows →
      value obs forwarded returned.losses row=
        ↑(Spec.loss cols o.logitScale o.smoothing o.squareScale (x row) (obs.integer initial labels row) o.ignored) ∧
      value obs forwarded returned.zLosses row=
        ↑(Spec.zLoss cols o.logitScale o.squareScale (x row) (obs.integer initial labels row) o.ignored) ∧
      ∀ col, col<cols → value obs final output (row*cols+col)=
        ↑(Spec.gradient cols o.logitScale o.smoothing o.squareScale (g row) (x row)
          (obs.integer initial labels row) o.ignored col) := by
  obtain ⟨e⟩ := forward_success RealSource.algebra rt o logits labels initial returned forwarded forwardCall
  have dims : e.rows=rows ∧ e.cols=cols := by
    have h := e.logitsShape.symm.trans shape
    simpa only [List.cons.injEq,and_true] using h
  have world : e.world=1 := by
    cases selected : o.processGroup with
    | none =>
      have call := e.sizeCall
      simp only [CrossEntropy1.Host.groupSize,selected,Step.ok.injEq] at call
      exact call.1.symm
    | some g =>
      have call : rt.worldSize g initial=.ok e.world e.sized := by
        simpa only [CrossEntropy1.Host.groupSize,selected] using e.sizeCall
      exact (groups.sizeCall _ _ _ _ call).1.trans (single g selected)
  have metadata := group_metadata obs rt groups o.processGroup initial e.sized e.ranked e.world e.rank e.sizeCall e.rankCall
  have rank : e.rank=0 := by omega
  have unsplit : ¬(e.world>1 ∨ e.cols>65536) := by omega
  obtain ⟨k⟩ := forward_kernel_data obs rt lib groups e logitsLive labelsLive strides
  have v := forward_view_data obs rt lib objects views k logitsLive strides registered
  have saved := unsplit_saved obs rt lib ints objects k v labelDtype unsplit
  have fv := returned_unsplit obs rt lib groups k dtype labelDtype unsplit smooth x
    (by simpa only [dims.1,dims.2] using caller)
    (by simpa only [world,Nat.one_mul,dims.1,dims.2] using targets)
  have options : returned.context.options=o := by simp only [e.returnIdentity]
  have total : returned.context.totalClasses=cols := by simp only [e.returnIdentity,world,Nat.one_mul,dims.2]
  have start : returned.context.classStart=0 := by simp only [e.returnIdentity,rank,Nat.zero_mul]
  have logitsShape := saved.logits.metadata.shape.trans shape
  have backwardSuccess := (backward_entry_success RealSource.algebra rt versions returned.context forwarded.native
    upstream output zUpstream backwardInitial final backwardCall).2
  obtain ⟨b⟩ := backward_success RealSource.algebra rt returned.context upstream output zUpstream backwardInitial final backwardSuccess
  have bdims : b.rows=rows ∧ b.cols=cols := by
    have h := b.shape.symm.trans logitsShape
    simpa only [List.cons.injEq,and_true] using h
  obtain ⟨bk⟩ := backward_kernel_data obs rt lib objects b (between.context saved.context) upstreamLive
  have lse : FloatArray obs forwarded returned.context.lse [b.rows]
      (fun row => ↑(Spec.lse returned.context.totalClasses returned.context.options.logitScale (x row))) := by
    refine ⟨by simpa only [dims.1,bdims.1] using saved.lseShape,saved.lseDtype,saved.context.lseLive,?_⟩
    intro row hr
    have bound : row<e.rows := by simpa only [Tensor.numel,saved.lseShape,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
    simpa only [total,options,dims.2] using (fv.2.2.2 row bound).2.1
  have savedLogits : ∀ row, row<b.rows → ∀ col, col<b.cols →
      value obs backwardInitial returned.context.logits (row*b.cols+col)=↑(x row (0+col)) := by
    intro row hr col hc
    have bound : row*b.cols+col<logits.numel := by
      simp only [Tensor.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
      have h := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt (show row<rows by omega))
      simp only [Nat.succ_mul] at h
      rw [bdims.2]
      omega
    exact (between.floatValue returned.context.logits (Or.inl rfl) _).trans
      ((saved.logits.floats dtype _ bound).trans (by simpa only [bdims.2,Nat.zero_add] using caller row (by omega) col (by omega)))
  have gradients := backward_real_values obs rt lib ints objects bk (between.context saved.context) saved.layout
    (saved.logits.metadata.dtype.trans dtype) saved.lseDense saved.labelsDense 0 start
    (by simpa only [options] using smooth) x (obs.integer initial labels) g savedLogits (between.lse lse)
    (by simpa only [dims.1,bdims.1] using between.labels saved.labels)
    (by simpa only [bdims.1] using upstreamShape) upstreamDtype
    (by simpa only [bdims.1] using upstreams)
  refine ⟨by simpa only [dims.1] using fv.1,by simpa only [dims.1] using fv.2.1,
    bk.outputShape.trans logitsShape,?_,?_⟩
  · intro inplace
    simpa only [options,inplace,↓reduceIte] using b.outputIdentity
  · intro row hr
    have f := fv.2.2.2 row (by omega)
    refine ⟨by simpa only [dims.2] using f.1,by simpa only [dims.2] using f.2.2,?_⟩
    intro col hc
    simpa only [bdims.2,total,options,Nat.zero_add] using gradients row (by omega) col (by omega)

#axiomsClean unsplit_forward_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
