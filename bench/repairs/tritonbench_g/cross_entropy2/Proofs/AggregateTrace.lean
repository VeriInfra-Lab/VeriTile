import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Trace

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind LocalTrace)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

structure AggregateData {α σ : Type} (rt : Runtime α σ) (o : Options) (count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Return) (final : World α σ) where
  pair : Tensor × Tensor
  reduced : World α σ
  localCalls : LocalTrace rt.toRuntime count losses lse initial pair reduced
  addedLoss : Tensor
  scaled : Tensor
  zLoss : Tensor
  correctedLoss : Tensor
  wideLabels : Tensor
  mask : Tensor
  added : World α σ
  scaledState : World α σ
  multiplied : World α σ
  corrected : World α σ
  widened : World α σ
  compared : World α σ
  masked : World α σ
  addCall : rt.addInplace pair.1 pair.2 reduced=.ok addedLoss added
  scaleCall : rt.scalarMul o.squareScale pair.2 added=.ok scaled scaledState
  multiplyCall : rt.multiply scaled pair.2 scaledState=.ok zLoss multiplied
  correctionCall : rt.addInplace addedLoss zLoss multiplied=.ok correctedLoss corrected
  widenCall : rt.toInt64 labels corrected=.ok wideLabels widened
  compareCall : rt.equalInt wideLabels o.ignored widened=.ok mask compared
  maskLossCall : rt.maskedFillZero correctedLoss mask compared=.ok returned.1 masked
  maskZCall : rt.maskedFillZero zLoss mask masked=.ok returned.2.2 final
  lseIdentity : returned.2.1=pair.2

theorem aggregate_success {α σ : Type} (rt : Runtime α σ) (o : Options) (count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Return) (final : World α σ)
    (success : aggregate rt o count labels losses lse initial=.ok returned final) :
    Nonempty (AggregateData rt o count labels losses lse initial returned final) := by
  unfold aggregate at success
  obtain ⟨pair,reduced,hlocal,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨addedLoss,added,hadd,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨scaled,scaledState,hscale,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨zLoss,multiplied,hmul,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨correctedLoss,corrected,hcorrect,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨wideLabels,widened,hwide,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨mask,compared,hcompare,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨maskedLoss,masked,hmask,success⟩ := (bind_success _ _ _ _).mp success
  obtain ⟨maskedZ,done,hmaskz,hreturn⟩ := (bind_success _ _ _ _).mp success
  cases hreturn
  have localCalls : LocalTrace rt.toRuntime count losses lse initial pair reduced := by
    by_cases many : count>1
    · rw [if_pos many] at hlocal
      obtain ⟨localLSE,logged,hlse,hlocal⟩ := (bind_success _ _ _ _).mp hlocal
      obtain ⟨localLoss,summed,hloss,hpair⟩ := (bind_success _ _ _ _).mp hlocal
      cases hpair
      exact .reduced many localLSE localLoss logged _ hlse hloss
    · rw [if_neg many] at hlocal
      cases hlocal
      exact .direct many
  exact ⟨⟨pair,reduced,localCalls,addedLoss,scaled,zLoss,correctedLoss,wideLabels,mask,
    added,scaledState,multiplied,corrected,widened,compared,masked,
    hadd,hscale,hmul,hcorrect,hwide,hcompare,hmask,hmaskz,rfl⟩⟩

#axiomsClean aggregate_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
