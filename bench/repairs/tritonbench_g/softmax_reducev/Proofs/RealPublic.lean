import bench.repairs.tritonbench_g.softmax_reducev.Proofs.RealSource
import bench.repairs.tritonbench_g.softmax_reducev.Proofs.Domain
import bench.repairs.tritonbench_g.softmax_reducev.Proofs.ABI
namespace VeriTile.Bench.Repairs.SoftmaxReduceV
open VeriTile.Triton FP.Structural
open Host
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- Independent logical attention: each batch's right-aligned page indices
select V[page,head,channel], and each logit comes from its own token window. -/
def mathematical (c : Call) (x : Input α) (logics values : Nat → ℝ) (j : Job) (d : Nat) : ℝ :=
  RealModel.target (x.counts j.batch) (fun k => logics (logicAddress c x j k))
    (fun k => values (valueAddress c x j k d))

structure FiniteInputs (c : Call) (x : Input (WithBot ℝ)) (logics values : Nat → ℝ) : Prop where
  logics : ∀ j, Legal c j → ∀ k, k < x.counts j.batch → x.logics (logicAddress c x j k)=↑(logics (logicAddress c x j k))
  values : ∀ j, Legal c j → ∀ k, k < x.counts j.batch → ∀ d, d < c.dimensions →
    x.values (valueAddress c x j k d)=↑(values (valueAddress c x j k d))

theorem real_inputs (c : Call) (x : Input (WithBot ℝ)) (logics values : Nat → ℝ)
    (finite : FiniteInputs c x logics values) (j : Job) (hj : Legal c j) :
    RealModel.Inputs (parameters c) (launch c x (fun _ => default) j)
      (fun k => logics (logicAddress c x j k))
      (fun k d => values (valueAddress c x j k d.val)) := by
  constructor
  · intro k hk
    have hk' : k < x.counts j.batch := by simpa [FPValue.sequenceLength,parameters,launch,Region.cast] using hk
    simpa [RealModel.keyLogicOffset,FPValue.sequenceStart,parameters,launch,Region.cast,logicAddress,Nat.add_assoc] using
      finite.logics j hj k hk'
  · intro k hk d
    have hk' : k < x.counts j.batch := by simpa [FPValue.sequenceLength,parameters,launch,Region.cast] using hk
    simpa [RealModel.keyValueOffset,FPValue.locationBase,FPValue.sequenceLength,parameters,launch,Region.cast,
      valueAddress,pageOffset,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      finite.values j hj k hk' d.val d.isLt

theorem real_row (c : Call) (x : Input (WithBot ℝ)) (valid : Admissible c x) (logics values : Nat → ℝ)
    (finite : FiniteInputs c x logics values) (j : Job) (hj : Legal c j) (d : Nat) (hd : d < c.dimensions) :
    row RealModel.model c x j d=↑(mathematical c x logics values j d) := by
  have result := RealModel.source_value (parameters c) (launch c x (fun _ => default) j)
    (fun k => logics (logicAddress c x j k)) (fun k d => values (valueAddress c x j k d.val))
    (real_inputs c x logics values finite j hj) (by change 0 < 64;decide)
    (by simpa [FPValue.sequenceLength,parameters,launch,Region.cast] using (valid.lengths j.batch hj.1).1)
    ((⟨d,hd⟩ : Fin c.dimensions),())
  simpa only [row,dif_pos hd,mathematical,FPValue.sequenceLength,parameters,launch,Region.cast,Cell.read_same,
    ↓reduceIte,reduceCtorEq] using result

/-- Every result channel of the actual complete (batch,head) launch satisfies
the independent weighted sum; all storage outside its output footprint is
preserved. Runtime success is only ordinary local source execution. -/
theorem correct_of_input (c : Call) (x : Input (WithBot ℝ)) (valid : Admissible c x)
    (logics values : Nat → ℝ) (finite : FiniteInputs c x logics values) (base : Nat)
    (invoke : Job → (Nat → WithBot ℝ) → SoftmaxTriton1.Result (Nat → WithBot ℝ) Unit)
    (rt : LocalRuntime RealModel.model c x invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial final : Memory (WithBot ℝ))
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions → final "Out" (base+address c j d)=↑(mathematical c x logics values j d)) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) := by
  have result := whole RealModel.model c x base invoke rt schedule complete initial final run
  exact ⟨fun j hj d hd => (result.1 j hj d hd).trans (real_row c x valid logics values finite j hj d hd),result.2⟩

/-- Public allocation-bound mathematical entry: every input value and metadata
index is decoded from the same initial caller memory whose complete frame is
proved. There is no independent input function supplied by the caller. -/
theorem correct (c : Call) (base : Nat) (initial final : Memory (WithBot ℝ))
    (valid : Admissible c (decodeMemory realIndex initial))
    (_metadata : RealMetadataEncoding c initial)
    (logics values : Nat → ℝ) (finite : FiniteInputs c (decodeMemory realIndex initial) logics values)
    (invoke : Job → (Nat → WithBot ℝ) → SoftmaxTriton1.Result (Nat → WithBot ℝ) Unit)
    (rt : AllocationRuntime c base initial (memoryEvaluate RealModel.model realIndex c base) invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c))
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions → final "Out" (base+address c j d)=
      ↑(mathematical c (decodeMemory realIndex initial) logics values j d)) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) :=
  correct_of_input c (decodeMemory realIndex initial) valid logics values finite base invoke
    (allocation_local RealModel.model realIndex c base initial invoke rt) schedule complete initial final run

#axiomsClean real_inputs
#axiomsClean real_row
#axiomsClean correct
#specNonCircular mathematical avoiding [row,FPValue.source,FPValue.update,FPValue.finalAccumulator]
end
end VeriTile.Bench.Repairs.SoftmaxReduceV
