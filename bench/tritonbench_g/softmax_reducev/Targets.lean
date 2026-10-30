import bench.tritonbench_g.softmax_reducev.IO
import bench.tritonbench_g.softmax_reducev.Proofs.RealMath

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

/-- Independent online operator recurrence on logical rows. This retains the
native reduction trees, least-finite clamp, padded lanes and live FMA sites. -/
def orderedStep (M : Algebra A) (D count : Nat)
    (q : Nat → A) (v : Nat → Fin D → A) (n : Nat) (old : FPValue.Accumulator A D) :
    FPValue.Accumulator A D :=
  let logits : Values A .real [64] := fun i =>
    M.cast (some .fp32) .fp32 .real (if n+i.1.val < count then q (n+i.1.val)
      else M.cast (some .fp32) .real .fp32 M.negInf)
  let vals : Values A .real [64,D] := fun i =>
    M.cast (some .fp32) .fp32 .real (if n+i.1.val < count then v (n+i.1.val) i.2.1
      else M.cast (some .fp32) .real .fp32 (M.literal (some .fp32) .real 0.0))
  let next := M.binary (some .fp32) .real .max
    (M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false logits ()) old.maximum
  let next := M.binary (some .fp32) .real .max next
    (M.literal (some .fp32) .real (-340282346638528859811704183484516925440.0))
  let scale := M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub old.maximum next)
  let weights : Values A .real [64] := fun i =>
    M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (logits i) next)
  let products : Values A .real [64,D] := fun i =>
    M.binary (some .fp32) .real .mul (weights (i.1,())) (vals i)
  { maximum := next
    normalizer := M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul old.normalizer scale)
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false weights ())
    output := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.output i) scale)
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false products i) }

def orderedRow (M : Algebra A) (D count : Nat) (q : Nat → A) (v : Nat → Fin D → A)
    (d : Fin D) : A :=
  let final := Range.foldRange (orderedStep M D count q v) 0 count 64
    ⟨M.negInf,M.literal none .real 0.0,fun _ => M.literal (some .fp32) .real 0⟩
  M.cast none .real .fp32 (M.binary (some .fp32) .real .div (final.output (d,())) final.normalizer)

def orderedTarget [Inhabited A] (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
    (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (counts : Fin c.batch → Nat) (xs : Inputs A c counts) : Outputs A c :=
  result fun b h d => observe b h d (orderedRow (M ⟨b.val,h.val⟩) c.dimensions (counts b)
    (fun k => encode (logitAt xs b h k)) (fun k d => encode (valueAt xs b h k d)) d)

/-- Finite gathered values and at least one finite logit in each active row.
Negative-infinity logits are admitted and carry zero mathematical mass. -/
def Normalizable {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs (WithBot ℝ) c counts) : Prop :=
  (∀ b h, ∃ k, k < counts b ∧ logitAt xs b h k ≠ ⊥) ∧
  (∀ b h k, k < counts b → ∀ d, valueAt xs b h k d ≠ ⊥)

def mathematical (c : Host.Call) (counts : Fin c.batch → Nat)
    (xs : Inputs (WithBot ℝ) c counts) : Outputs (WithBot ℝ) c :=
  result fun b h d => ↑(RealModel.target (counts b) (logitAt xs b h)
    (fun k => (valueAt xs b h k d).unbotD 0))

end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
