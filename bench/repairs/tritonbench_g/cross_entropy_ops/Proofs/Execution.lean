import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Source

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Source
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Memory Cell write)
noncomputable section
set_option maxHeartbeats 4000000

theorem forwardLoss_input {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (before after : Memory α) (same : ∀ i, after p.logits i=before p.logits i)
    (row block : Nat) (target : Int) (values : CrossEntropy1.Source.ForwardValues α p.block) :
    forwardLoss M p after row block target values=forwardLoss M p before row block target values := by
  apply CrossEntropy1.Program.forwardLoss_input
  intro i
  unfold scaledReads
  rw [same i]

/-- All three physical scalar stores are read back, including the independent
logging z-loss. The target gather is protected by fresh LSE allocation. -/
theorem forward_outputs {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat)
    (lseLoss : p.lse≠p.loss) (lseLogits : p.lse≠p.logits)
    (zLossNe : p.zloss≠p.loss) (zLse : p.zloss≠p.lse) :
    (forward M p memory row block p.loss (p.lossBase+block*p.rows+row)).read .real=
      forwardLoss M p memory row block (CrossEntropy1.Source.label p.toParameters memory row)
        (forwardValues M p memory row block) ∧
    (forward M p memory row block p.lse (p.lseBase+block*p.rows+row)).read .real=
      (forwardValues M p memory row block).lse ∧
    (p.split=Bool.false →
      (forward M p memory row block p.zloss (p.zlossBase+block*p.rows+row)).read .real=
        zLoss M p (CrossEntropy1.Source.label p.toParameters memory row)
          (forwardValues M p memory row block).lse) := by
  have stable : ∀ i, write memory p.lse (p.lseBase+block*p.rows+row)
      (.mk .real (forwardValues M p memory row block).lse) p.logits i=memory p.logits i := by
    intro i
    exact CrossEntropy1.Program.write_other _ _ _ _ _ _ (Or.inl lseLogits.symm)
  have lossEq := forwardLoss_input M p memory _ stable row block
    (CrossEntropy1.Source.label p.toParameters memory row) (forwardValues M p memory row block)
  simp only [forward]
  rw [lossEq]
  by_cases split : p.split=Bool.true
  · simp [split,write,Cell.read_same,lseLoss,lseLoss.symm]
  · have unsplit : p.split=Bool.false := Bool.eq_false_iff.mpr split
    simp [unsplit,write,Cell.read_same,lseLoss,lseLoss.symm,zLossNe,zLossNe.symm,zLse,zLse.symm]

/-- An inactive scaled padding word is discarded by the new post-scale mask.
This is an exact selector law for arbitrary word algebras, including NaNs. -/
theorem masked_scaled_padding {α : Type} (M : FP32 α) (scale raw : α) :
    (if (Bool.false : Bool) then M.mul raw scale else M.negInf)=M.negInf := rfl

#axiomsClean forwardLoss_input
#axiomsClean forward_outputs
#axiomsClean masked_scaled_padding
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Source
