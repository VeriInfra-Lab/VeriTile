import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Targets
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.NativeGraph

/-! The former independent recurrence had precision scopes inherited from
extra source casts. The pasted body has separate storage-padding conversions
and a materialized FP32 zero initializer. These exact concrete model equalities
retain every input word, shape, count, native reduction plan and Real value;
no arbitrary cast identity or additional numerical premise is introduced. -/
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PrecisionProfile
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def previousStep (M : Algebra A) (D count : Nat)
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

def previousRow (M : Algebra A) (D count : Nat) (q : Nat → A) (v : Nat → Fin D → A)
    (d : Fin D) : A :=
  let final := Range.foldRange (previousStep M D count q v) 0 count 64
    ⟨M.negInf,M.literal none .real 0.0,fun _ => M.literal (some .fp32) .real 0⟩
  M.cast none .real .fp32 (M.binary (some .fp32) .real .div (final.output (d,())) final.normalizer)


theorem native_target_unchanged (plan : NativeGraph.Plan) (D count : Nat)
    (q : Nat → NativeGraph.Node) (v : Nat → Fin D → NativeGraph.Node) (d : Fin D) :
    previousRow (NativeGraph.algebra plan) D count q v d =
      PublicIO.orderedRow (NativeGraph.algebra plan) D count q v d := rfl

theorem real_target_unchanged (D count : Nat)
    (q : Nat → WithBot ℝ) (v : Nat → Fin D → WithBot ℝ) (d : Fin D) :
    previousRow RealModel.model D count q v d =
      PublicIO.orderedRow RealModel.model D count q v d := rfl

#axiomsClean native_target_unchanged
#axiomsClean real_target_unchanged
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PrecisionProfile
