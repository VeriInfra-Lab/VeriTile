import bench.repairs.tritonbench_g.cross_entropy_ops.Contract
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Entries

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Checked
open VeriTile.Triton Host
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Step Observations)
open CrossEntropy1.TensorModel (Tensor)
noncomputable section

 theorem forward_success {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {m : ForwardMetadata} {scaleFloat scaleFinite : Bool} {o : Options} {logits labels : Tensor}
    {target : Nat → Int} {initial final : World α σ} {returned : ForwardReturn} :
    forward M rt m scaleFloat scaleFinite o logits labels target initial=.ok returned final ↔
      InputContract m scaleFloat scaleFinite o logits labels target ∧ Host.forward M rt o logits labels initial=.ok returned final := by
  classical
  unfold forward
  split_ifs with h <;> simp [h]

theorem backward_success {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {versions : σ → Nat → Nat} {c : Context} {saved : σ} {m : BackwardMetadata} {rows : Nat}
    {upstream output : Tensor} {zUpstream : Option Tensor} {initial final : World α σ} :
    backward M rt versions c saved m rows upstream zUpstream initial=.ok output final ↔
      BackwardContract m rows c.logits upstream ∧ Host.backwardEntry M rt versions c saved upstream zUpstream initial=.ok output final := by
  classical
  by_cases same : CrossEntropy1.Autograd.Unmodified versions
      (CrossEntropy1.Autograd.capture versions c.base saved) initial
  · simp only [backward,CrossEntropy1.Autograd.retrieve,if_pos same,CrossEntropy1.Host.bind]
    split_ifs with h <;> simp [h]
  · simp [backward,CrossEntropy1.Autograd.retrieve,same,CrossEntropy1.Host.bind,Host.backwardEntry]

 theorem source_success {α σ : Type} [Inhabited α] {M : FP32 α} {obs : Observations α σ}
    (call : Entry.Invocation obs M) (m : ForwardMetadata) (scaleFloat scaleFinite : Bool)
    (h : InputContract m scaleFloat scaleFinite call.options call.logits call.labels (obs.integer call.initial call.labels)) :
    forward M call.runtime m scaleFloat scaleFinite call.options call.logits call.labels
      (obs.integer call.initial call.labels) call.initial=.ok call.returned call.saved :=
  forward_success.mpr ⟨h,call.success⟩

 theorem invocation_inputs {α σ : Type} [Inhabited α] {M : FP32 α} {obs : Observations α σ}
    (call : Entry.Invocation obs M) (m : ForwardMetadata) (scaleFloat scaleFinite : Bool)
    (h : InputContract m scaleFloat scaleFinite call.options call.logits call.labels (obs.integer call.initial call.labels)) :
    call.data.rows=m.rows ∧ call.data.cols=m.cols ∧ call.data.world=1 ∧ call.data.rank=0 ∧
    call.logits.dtype="torch.float32" ∧ call.labels.dtype="torch.int64" ∧ 0≤call.options.smoothing ∧
    call.options.processGroup=none ∧
    (∀ row, row<call.data.rows → obs.integer call.initial call.labels row=call.options.ignored ∨
      (0≤obs.integer call.initial call.labels row ∧ obs.integer call.initial call.labels row<(call.data.cols : Int))) := by
  have contract := h.1
  have shape : call.logits.shape=[m.rows,m.cols] := by unfold CrossEntropy1.Checked.InputContract at contract; tauto
  have group : call.options.processGroup=none := by unfold CrossEntropy1.Checked.InputContract at contract; tauto
  have shapes := call.data.logitsShape.symm.trans shape
  have rows : call.data.rows=m.rows := (List.cons.inj shapes).1
  have cols : call.data.cols=m.cols := (List.cons.inj (List.cons.inj shapes).2).1
  have world : call.data.world=1 := by
    have size := call.data.sizeCall
    simp only [CrossEntropy1.Host.groupSize,group,Step.ok.injEq] at size
    exact size.1.symm
  have rank : call.data.rank=0 := by
    have rank := call.data.rankCall
    simp only [CrossEntropy1.Host.groupRank,group,Step.ok.injEq] at rank
    exact rank.1.symm
  refine ⟨rows,cols,world,rank,?_,?_,?_,group,?_⟩
  · unfold CrossEntropy1.Checked.InputContract at contract; tauto
  · unfold CrossEntropy1.Checked.InputContract at contract; tauto
  · unfold CrossEntropy1.Checked.InputContract at contract; tauto
  · rw [rows,cols]
    rcases contract with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,targets⟩
    exact targets

#axiomsClean forward_success
#axiomsClean backward_success
#axiomsClean source_success
#axiomsClean invocation_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Checked
