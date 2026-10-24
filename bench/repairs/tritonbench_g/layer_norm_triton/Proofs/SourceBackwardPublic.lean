import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardReal
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceSaved

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardPublic
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section
variable {x w dy r mu : Input}

theorem real_executes (model : SourceModel.Model ℝ) (laws : SourceFinalReal.Laws model) (B : Nat)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (hb : N≤B) :
    ∃ result,SourceBackwardStorage.exec model B Memory.realAPI Neg.neg c memory=some result ∧
      result=SourceBackwardStorage.final model B Memory.realAPI Neg.neg c memory ∧
      (∀ k i,logical Memory.realAPI Neg.neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then KernelBackward.realValues c memory k i
        else logical Memory.realAPI Neg.neg memory (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg result (c.first.buffers.output k).tensor i.val=
          SourceBackwardGroupReal.values c memory k i) ∧
      (∀ key,KernelBackward.Outside c key → observe Memory.realAPI result key=observe Memory.realAPI memory key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe Memory.realAPI result key=observe Memory.realAPI memory key) := by
  obtain ⟨result,he,hm,hv,hz,hs,ho,hf⟩ := SourceBackwardResult.executes model B Memory.realAPI Neg.neg c memory
  refine ⟨result,he,hm,?_,hz,?_,ho,hf⟩
  · intro k i
    rw [hv]
    by_cases hl : KernelContract.launched M N=Bool.true
    · simp only [hl,if_true,SourceBackwardReal.values_real model laws B c memory
        ((KernelContract.launched_iff M N).mp hl).2 hb]
    · simp [hl]
  · intro hl k i
    rw [hs hl,SourceBackwardGroupReal.values_real model laws.toLaws B c memory
      ((KernelContract.launched_iff M N).mp hl).2 hb]

/-- Constructive sequential composition: backward reads the actual storage
produced by forward, and saved statistics are derived rather than assumed. -/
theorem composed (model : SourceModel.Model ℝ) (laws : SourceFinalReal.Laws model) (B : Nat) (eps : ℝ)
    (f : KernelContract.Forward M N) (b : KernelContract.Backward M N f.x f.w dy r mu)
    (rBinding : r.dense=f.buffers.output 0) (muBinding : mu.dense=f.buffers.output 1)
    (memory : World (Nat → ℝ)) (hb : N≤B) :
    ∃ forward backward,
      SourceForwardStorage.exec model B eps Memory.realAPI Neg.neg f memory=some forward ∧
      SourceBackwardStorage.exec model B Memory.realAPI Neg.neg b forward=some backward ∧
      forward=SourceForwardStorage.final model B eps Memory.realAPI Neg.neg f memory ∧
      backward=SourceBackwardStorage.final model B Memory.realAPI Neg.neg b forward ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical Memory.realAPI Neg.neg forward (f.buffers.output k).tensor i.val=KernelForward.realValues f eps memory k i) ∧
      (∀ k i,logical Memory.realAPI Neg.neg backward (KernelBackward.outputs b k).tensor i.val=
        if KernelContract.launched M N then KernelBackward.realValues b forward k i
        else logical Memory.realAPI Neg.neg forward (KernelBackward.outputs b k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.true → ∀ row,row<M →
        KernelBackward.initialStats Memory.realAPI Neg.neg r mu forward row=
          ⟨RealProof.Spec.mean N (KernelForward.row Memory.realAPI Neg.neg f.x memory N row),
           RealProof.Spec.inverse N (KernelForward.row Memory.realAPI Neg.neg f.x memory N row) eps⟩) ∧
      (KernelContract.launched M N=Bool.false → forward=memory ∧ backward=memory) ∧
      (∀ key,Outside f.buffers key → KernelBackward.Outside b key →
        observe Memory.realAPI backward key=observe Memory.realAPI memory key) := by
  obtain ⟨forward,hfe,hfm,hfv,hfz,hfo,hff⟩ := SourceForwardPublic.real_executes model laws.toLaws B eps f memory hb
  obtain ⟨backward,hbe,hbm,hbv,hbz,hbs,hbo,hbf⟩ := real_executes model laws B b forward hb
  refine ⟨forward,backward,hfe,hbe,hfm,hbm,hfv,hbv,?_,?_,?_⟩
  · intro hl row hr
    rw [hfm]
    exact SourceSaved.saved_real model laws.toLaws B eps f memory r mu rBinding muBinding hl hb row hr
  · intro hl
    exact ⟨hfz hl,(hbz hl).trans (hfz hl)⟩
  · intro key hf hb
    exact (hbo key hb).trans (hfo key hf)

#axiomsClean real_executes
#axiomsClean composed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardPublic
