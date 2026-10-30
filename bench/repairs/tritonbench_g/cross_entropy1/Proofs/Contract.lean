import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Native
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.HostContract
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Entries

namespace VeriTile.Bench.Repairs.CrossEntropy1.Checked
open VeriTile.Triton Source Host
open TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

@[simp] theorem forward_admission (m : ForwardMetadata) (o : Options) (logits labels : Tensor) (target : Nat → Int) :
    checkForward m o logits labels target=Bool.true ↔ InputContract m o logits labels target := by
  simp [checkForward]

@[simp] theorem backward_admission (m : BackwardMetadata) (rows : Nat) (logits upstream : Tensor) :
    checkBackward m rows logits upstream=Bool.true ↔ BackwardContract m rows logits upstream := by
  simp [checkBackward]

theorem source_success {α σ : Type} [Inhabited α] {M : FP32 α} {obs : Observations α σ}
    (call : Entry.Invocation obs M) (m : ForwardMetadata)
    (h : InputContract m call.options call.logits call.labels (obs.integer call.initial call.labels)) :
    forward M call.runtime m call.options call.logits call.labels (obs.integer call.initial call.labels) call.initial=
      .ok call.returned call.saved := by
  simpa only [forward,(forward_admission _ _ _ _ _).mpr h,ite_true] using call.success

/-- The local world/rank and all Real input premises are derived from guards
and the actual wrapper shape/group queries. They are not extra admissions. -/
theorem invocation_inputs {α σ : Type} [Inhabited α] {M : FP32 α} {obs : Observations α σ}
    (call : Entry.Invocation obs M) (m : ForwardMetadata)
    (h : InputContract m call.options call.logits call.labels (obs.integer call.initial call.labels)) :
    call.data.rows=m.rows ∧ call.data.cols=m.cols ∧ call.data.world=1 ∧ call.data.rank=0 ∧
    call.logits.dtype="torch.float32" ∧ call.labels.dtype="torch.int64" ∧ 0≤call.options.smoothing ∧
    (∀ row, row<call.data.rows → obs.integer call.initial call.labels row=call.options.ignored ∨
      (0≤obs.integer call.initial call.labels row ∧ obs.integer call.initial call.labels row<(call.data.world*call.data.cols : Nat))) := by
  have shape : call.logits.shape=[m.rows,m.cols] := by unfold InputContract at h; tauto
  have group : call.options.processGroup=none := by unfold InputContract at h; tauto
  have shapes := call.data.logitsShape.symm.trans shape
  have rows : call.data.rows=m.rows := by simpa only [List.cons.injEq] using (List.cons.inj shapes).1
  have cols : call.data.cols=m.cols := by simpa only [List.cons.injEq] using (List.cons.inj (List.cons.inj shapes).2).1
  have world : call.data.world=1 := by
    have size := call.data.sizeCall
    simp only [groupSize,group,Step.ok.injEq] at size
    exact size.1.symm
  have rank : call.data.rank=0 := by
    have rank := call.data.rankCall
    simp only [groupRank,group,Step.ok.injEq] at rank
    exact rank.1.symm
  refine ⟨rows,cols,world,rank,?_,?_,?_,?_⟩
  · unfold InputContract at h; tauto
  · unfold InputContract at h; tauto
  · unfold InputContract at h; tauto
  · rw [rows,cols,world,Nat.one_mul]
    rcases h with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,targets⟩
    exact targets


theorem backward_source_success {α σ : Type} [Inhabited α] {M : FP32 α} {obs : Observations α σ}
    (call : Entry.Invocation obs M) (versions : σ → Nat → Nat) (before after : World α σ)
    (upstream output : Tensor) (m : BackwardMetadata)
    (h : BackwardContract m call.data.rows call.returned.context.logits upstream)
    (success : Autograd.backwardEntry M call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after) :
    backward M call.runtime versions (Autograd.capture versions call.returned.context call.saved.native)
      m call.data.rows upstream before=.ok output after := by
  have result := Autograd.backward_success M call.runtime versions _ upstream output before after success
  simp only [backward,Autograd.retrieve,if_pos result.1,Host.bind]
  simpa only [Autograd.capture_context,(backward_admission _ _ _ _).mpr h,ite_true] using result.2


theorem rank_inputs_single {α σ : Type} [Inhabited α] {obs : Observations α σ} {A : TensorArithmetic α}
    {rt : Runtime α σ} {gs gr : Nat → Nat} (comms : CollectiveLibrary obs A rt gs gr)
    {o : Options} {world rows : Nat} {localPair : Tensor × Tensor} {initial : World α σ}
    {pair : Tensor × Tensor} {final : World α σ}
    (trace : RankTrace rt o world rows localPair initial pair final) (lossPeers lsePeers : Nat → α)
    (one : world≤1) : RankInputs comms trace lossPeers lsePeers := by
  cases trace with
  | direct => trivial
  | reduced many => omega

theorem blocks_antitone (n small large : Nat) (positive : 0<small) (widths : small≤large) :
    Math.blocks n large≤Math.blocks n small := by
  have cover := Math.blocks_cover n small positive
  have upper := Nat.mul_le_mul_left (Math.blocks n small) widths
  have larger : 0<large := lt_of_lt_of_le positive widths
  unfold Math.blocks at *
  apply (Nat.div_le_iff_le_mul larger).mpr
  omega

/-- Explicit guards imply both launch bounds and every finite column mask
bound used by the retained signed-32 PID to signed-64 source proof. -/
theorem native_launches (m : ForwardMetadata) (o : Options) (logits labels : Tensor) (targets : Nat → Int)
    (h : InputContract m o logits labels targets) :
    Native.Supported m.rows (Math.blocks m.cols (fwdWidth m.cols)) (fwdWidth m.cols) m.cols ∧
    Native.Supported m.rows (Math.blocks m.cols (bwdWidth m.cols)) (bwdWidth m.cols) m.cols := by
  have positive : 0<m.cols := by unfold InputContract at h; tauto
  have rows : m.rows≤2^31-1 := by unfold InputContract at h; tauto
  have blocks : Math.blocks m.cols (bwdWidth m.cols)≤65535 := by unfold InputContract at h; tauto
  have bw := backward_width m.cols positive
  have fw := forward_width m.cols positive
  have widths : bwdWidth m.cols≤fwdWidth m.cols := by
    unfold bwdWidth fwdWidth
    omega
  have forwardBlocks := (blocks_antitone m.cols _ _ bw.1 widths).trans blocks
  have cover := Math.blocks_cover m.cols (bwdWidth m.cols) bw.1
  have limit := Nat.mul_le_mul blocks bw.2
  have cols : m.cols<2^63 := by norm_num at *; omega
  exact ⟨⟨rows,forwardBlocks,fw.1,fw.2.1,cols⟩,⟨rows,blocks,bw.1,by omega,cols⟩⟩

#axiomsClean rank_inputs_single
#axiomsClean native_launches
#axiomsClean backward_source_success
#axiomsClean forward_admission
#axiomsClean backward_admission
#axiomsClean source_success
#axiomsClean invocation_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Checked
