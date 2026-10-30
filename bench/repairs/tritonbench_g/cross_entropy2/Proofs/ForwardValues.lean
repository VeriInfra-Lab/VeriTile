import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardKernel
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AggregationValues

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations Preserves FloatArray IntegerViews value localResult)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

variable {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options}
  {x y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Return}
  {e : ForwardData M rt o x y rows cols initial returned final}

/-- The complete grid preserves every region that was live before the fresh
output allocations, including all bytes beyond the caller's logical views. -/
theorem forward_prepared_frame (ints : IntegerViews obs) (k : ForwardKernelData obs e) :
    Preserves obs e.normalizedLabels e.computed := by
  have lossFresh := k.lossAllocation.fresh
  have lseFresh : ¬e.normalizedLabels.live e.lse.region := fun live => k.lseAllocation.fresh (k.lossAllocation.frame.live _ live)
  have zFresh : ¬e.normalizedLabels.live e.zLoss.region := fun live =>
    k.zAllocation.fresh (k.lseAllocation.frame.live _ (k.lossAllocation.frame.live _ live))
  have unchanged : ∀ r, e.normalizedLabels.live r → ∀ i,
      Grid.forward M e.parameters k.ready.memory e.count r i=k.ready.memory r i := by
    intro r live i
    apply Grid.complete_frame M e.parameters k.ready.memory e.count k.separation
    intro q hq writes
    rcases writes with ⟨eq,_⟩|⟨eq,_⟩|⟨eq,_⟩
    · exact lossFresh (by simpa only [ForwardData.parameters,Host.parameters] using (eq ▸ live : e.normalizedLabels.live e.parameters.loss))
    · exact lseFresh (by simpa only [ForwardData.parameters,Host.parameters] using (eq ▸ live : e.normalizedLabels.live e.parameters.lse))
    · exact zFresh (by simpa only [ForwardData.parameters,Host.parameters] using (eq ▸ live : e.normalizedLabels.live e.parameters.zLoss))
  rw [k.computedEq]
  refine ⟨k.labelsFrame.live,?_,?_⟩
  · intro r i live
    exact (unchanged r live i).trans (k.labelsFrame.memory r i live)
  · intro t i live
    exact (ints.localRead {k.ready with memory:=Grid.forward M e.parameters k.ready.memory e.count}
      k.ready t i rfl (unchanged t.region live)).trans (k.labelsFrame.integers t i live)

theorem forward_label_array (ints : IntegerViews obs) (k : ForwardKernelData obs e)
    (yLive : initial.live y.region) :
    IntegerArray obs "torch.int32" e.computed e.labels [rows] (obs.integer initial y) := by
  have normalized : IntegerArray obs "torch.int32" e.normalizedLabels e.labels [rows] (obs.integer initial y) := by
    refine ⟨k.labels.materialized.metadata.shape.trans e.admitted.2.2.2.1,
      k.labels.materialized.metadata.dtype.trans e.admitted.2.2.1,k.labels.materialized.live,?_⟩
    intro i hi
    exact (k.labels.int32 e.admitted.2.2.1 i (by simpa only [k.labels.materialized.metadata.numel] using hi)).trans
      (k.logits.materialized.frame.integers y i yLive)
  exact normalized.preserve (forward_prepared_frame ints k)

def finalLoss (k : ForwardKernelData obs e) (A : Arithmetic α) : Nat → α :=
  if e.split then aggregateLossValue A o
    (localResult A.sum e.count rows (k.outputValues .loss))
    (localResult A.logsumexp e.count rows (k.outputValues .lse)) (obs.integer initial y)
  else k.outputValues .loss
def finalLSE (k : ForwardKernelData obs e) (A : Arithmetic α) : Nat → α :=
  if e.split then localResult A.logsumexp e.count rows (k.outputValues .lse) else k.outputValues .lse
def finalZ (k : ForwardKernelData obs e) (A : Arithmetic α) : Nat → α :=
  if e.split then aggregateZValue A o (localResult A.logsumexp e.count rows (k.outputValues .lse)) (obs.integer initial y)
  else k.outputValues .zLoss

/-- Complete forward host readback. Every value comes from a source store and
the subsequently executed local library calls, with no final-value premise. -/
theorem forward_values (A : Arithmetic α) (lib : Library obs A rt) (ints : IntegerViews obs)
    (k : ForwardKernelData obs e) (yLive : initial.live y.region) :
    FloatArray obs final returned.1 [rows] (finalLoss k A) ∧
    FloatArray obs final returned.2.1 [rows] (finalLSE k A) ∧
    FloatArray obs final returned.2.2 [rows] (finalZ k A) ∧
    IntegerArray obs "torch.int32" final e.labels [rows] (obs.integer initial y) ∧
    returned.1.region≠returned.2.1.region ∧ returned.1.region≠returned.2.2.region ∧
    returned.2.1.region≠returned.2.2.region ∧ Preserves obs initial final := by
  have loss := forward_arrays k .loss
  have lse := forward_arrays k .lse
  have z := forward_arrays k .zLoss
  have labels := forward_label_array ints k yLive
  have preparedFrame := forward_prepared_frame ints k
  have initialFrame := (k.logits.materialized.frame.trans k.labels.materialized.frame).trans preparedFrame
  have ls : e.loss.region≠e.lse.region := by
    intro h; have := k.separation.distinct (show Grid.region e.parameters .loss=Grid.region e.parameters .lse from h)
    cases this
  have lz : e.loss.region≠e.zLoss.region := by
    intro h; have := k.separation.distinct (show Grid.region e.parameters .loss=Grid.region e.parameters .zLoss from h)
    cases this
  have sz : e.lse.region≠e.zLoss.region := by
    intro h; have := k.separation.distinct (show Grid.region e.parameters .lse=Grid.region e.parameters .zLoss from h)
    cases this
  have finish := e.finishCall
  cases split : e.split with
  | «false» =>
    simp only [split,Bool.false_eq_true,↓reduceIte,Step.ok.injEq] at finish
    have count : e.count=1 := by
      have pos : 0<e.count := by
        rw [e.countEq]
        exact VeriTile.Bench.Repairs.CrossEntropy1.Host.block_count_positive cols o.width e.admitted.1 e.admitted.2.2.2.2.2.1
      have flag := e.splitEq
      rw [split] at flag
      have small : ¬e.count>1 := by
        intro many
        simp only [decide_eq_true (show e.count>1 ∨ o.start≠0 ∨ cols≠o.total from Or.inl many),Bool.or_true] at flag
        cases flag
      omega
    have values : returned=(e.loss,e.lse,e.zLoss) := finish.1.symm
    refine ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩
    · simpa only [values,← finish.2,finalLoss,split,Bool.false_eq_true,↓reduceIte,count,Nat.lt_irrefl] using loss
    · simpa only [values,← finish.2,finalLSE,split,Bool.false_eq_true,↓reduceIte,count,Nat.lt_irrefl] using lse
    · simpa only [values,← finish.2,finalZ,split,Bool.false_eq_true,↓reduceIte,count,Nat.lt_irrefl] using z
    · simpa only [← finish.2] using labels
    · simpa only [values] using ls
    · simpa only [values] using lz
    · simpa only [values] using sz
    · simpa only [← finish.2] using initialFrame
  | «true» =>
    simp only [split,↓reduceIte] at finish
    obtain ⟨trace⟩ := aggregate_success rt o e.count e.labels e.loss e.lse e.computed returned final finish
    obtain ⟨idLoss,idZ,lossFinal,lseFinal,zFinal,labelsFinal,lsFinal,lzFinal,szFinal,frame⟩ :=
      aggregation_values obs A rt lib o rows e.count e.labels e.loss e.lse e.computed returned final
        (k.outputValues .loss) (k.outputValues .lse) (obs.integer initial y) loss lse labels ls
        (k.separation.labels .loss) trace
    refine ⟨?_,?_,?_,labelsFinal,lsFinal,lzFinal,szFinal,?_⟩
    · simpa only [finalLoss,split,↓reduceIte] using lossFinal
    · simpa only [finalLSE,split,↓reduceIte] using lseFinal
    · simpa only [finalZ,split,↓reduceIte] using zFinal
    · exact frame initial initialFrame (fun live => k.lossAllocation.fresh
        (k.labels.materialized.frame.live _ (k.logits.materialized.frame.live _ live)))

#axiomsClean forward_prepared_frame
#axiomsClean forward_label_array
#axiomsClean forward_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
