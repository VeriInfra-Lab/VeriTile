import bench.tritonbench_g.cross_entropy1.Proofs.Correction

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

structure IntArray {α σ : Type} (obs : Observations α σ)
    (s : World α σ) (t : Tensor) (shape : List Nat) (values : Nat → Int) : Prop where
  shapeEq : t.shape=shape
  dtype : t.dtype="torch.int64"
  live : s.live t.region
  values : ∀ i, i<t.numel → obs.integer s t i=values i

theorem IntArray.preserve {α σ : Type} {obs : Observations α σ} {a b : World α σ}
    {t : Tensor} {shape : List Nat} {v : Nat → Int} (array : IntArray obs a t shape v)
    (frame : Preserves obs a b) : IntArray obs b t shape v :=
  ⟨array.shapeEq,array.dtype,frame.live _ array.live,
   fun i hi => (frame.integers t i array.live).trans (array.values i hi)⟩

theorem IntArray.mutate {α σ : Type} {obs : Observations α σ} {a b : World α σ}
    {t dst : Tensor} {shape : List Nat} {v : Nat → Int} (array : IntArray obs a t shape v)
    (frame : Mutation obs a b dst) (separate : t.region≠dst.region) : IntArray obs b t shape v :=
  ⟨array.shapeEq,array.dtype,frame.live _ array.live,
   fun i hi => (frame.integers t i array.live separate).trans (array.values i hi)⟩

/-- Ordinary observation rules. The dense int64 rule is used only after the
fresh label snapshot. The locality rule also covers caller integer views that
share backing bytes with FP32 views. It does not assume any kernel output. -/
structure IntegerViews {α σ : Type} [Inhabited α] (obs : Observations α σ) : Prop where
  dense : ∀ s t i, t.dtype="torch.int64" → t.Dense → i<t.numel →
    obs.integer s t i=(s.memory t.region i).read .int
  localRead : ∀ before after t i, before.native=after.native →
    (∀ index, before.memory t.region index=after.memory t.region index) →
    obs.integer before t i=obs.integer after t i

theorem mask_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt)
    (rows : Nat) (ignored : Int) (losses labels mask output : Tensor)
    (initial compared final : World α σ) (lv : Nat → α) (targets : Nat → Int)
    (loss : FloatArray obs initial losses [rows] lv) (label : IntArray obs initial labels [rows] targets)
    (compareCall : rt.equalInt labels ignored initial=.ok mask compared)
    (maskCall : rt.maskedFillZero losses mask compared=.ok output final) :
    output=losses ∧ Mutation obs initial final losses ∧
      FloatArray obs final output [rows] (fun row => if targets row=ignored then A.zero else lv row) := by
  have comparison := ops.equalInt labels ignored initial mask compared label.live label.dtype compareCall
  have lossCompared := loss.preserve comparison.1.frame
  have sameShape : losses.shape=mask.shape := loss.shapeEq.trans (comparison.1.shapeEq.trans label.shapeEq).symm
  have masked := ops.maskedFill losses mask compared output final lossCompared.live comparison.1.live
    loss.dtype comparison.1.dtypeEq sameShape maskCall
  refine ⟨masked.1,masked.2.1.prepend comparison.1.frame,?_⟩
  rw [masked.1]
  refine ⟨loss.shapeEq,loss.dtype,masked.2.1.live _ lossCompared.live,?_⟩
  intro row hr
  have labelBound : row<labels.numel := by simpa only [Tensor.numel,label.shapeEq,loss.shapeEq] using hr
  have maskBound : row<mask.numel := by simpa only [Tensor.numel,← sameShape] using hr
  rw [masked.2.2 row hr,comparison.2.1.2 row maskBound,comparison.2.2 row labelBound,
    label.values row labelBound,lossCompared.values row hr]
  simp only [decide_eq_true_eq]

#axiomsClean mask_values
#axiomsClean IntArray.preserve
#axiomsClean IntArray.mutate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
