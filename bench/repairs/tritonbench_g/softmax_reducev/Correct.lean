import bench.repairs.tritonbench_g.softmax_reducev.Proofs.RealPublic
import bench.repairs.tritonbench_g.softmax_reducev.Proofs.Contents

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.Correct
open VeriTile.Triton FP.Structural Checked
noncomputable section

def realEvaluate (c : TensorCall) (m : Host.Memory (WithBot ℝ)) :=
  Host.evaluate RealModel.model (host c.observation) (Host.decodeMemory Host.realIndex (view c m))
def invokeReal (c : TensorCall) (work : List Host.Job) (m : Host.Memory (WithBot ℝ)) :=
  if checkInput c.observation (realObserved (view c m) "Starts")
      (realObserved (view c m) "Lengths") (realObserved (view c m) "Locations") = .ok () then
    invoke c work (realEvaluate c m) m else .error "input contract rejected" m

def realTarget (c : TensorCall) (m : Host.Memory (WithBot ℝ)) (logics values : Nat → ℝ)
    (j : Host.Job) (d : Nat) :=
  mathematical (host c.observation) (Host.decodeMemory Host.realIndex (view c m)) logics values j d

theorem tensor_correct (c : TensorCall)
    (m : Host.Memory (WithBot ℝ))
    (h : InputContract c.observation (realObserved (view c m) "Starts")
      (realObserved (view c m) "Lengths") (realObserved (view c m) "Locations"))
    (metadata : Host.RealMetadataEncoding (host c.observation) (view c m))
    (logics values : Nat → ℝ)
    (finite : FiniteInputs (host c.observation) (Host.decodeMemory Host.realIndex (view c m)) logics values)
    (work : List Host.Job) (complete : work.Perm (Host.workset (host c.observation))) :
    ∃ result,invokeReal c work m=.ok result result.memory ∧ result.returned=none ∧
      result.output=c.output.tensor ∧
      (∀ j,Host.Legal (host c.observation) j → ∀ d,d < dimensions c.observation →
        result.memory c.output.region (c.output.tensor.storageOffset+Host.address (host c.observation) j d)=
          ↑(realTarget c m logics values j d)) ∧
      (∀ r i,Exterior c r i → result.memory r i=m r i) := by
  let source : Host.SourceLawGeneral (host c.observation) (realEvaluate c m)
      (Host.row RealModel.model (host c.observation) (Host.decodeMemory Host.realIndex (view c m))) :=
    ⟨Host.source_law RealModel.model _ _⟩
  obtain ⟨final,run,results,frame⟩ := source_whole (host c.observation) (realEvaluate c m) _ source
    (Host.source_total RealModel.model _ _) work complete (view c m)
  refine ⟨⟨c.output.tensor,none,physical c m final⟩,?_,rfl,rfl,?_,physical_frame c m final frame⟩
  · simp [invokeReal,(check_input_iff _ _ _ _).mpr h,invoke,run,liftResult]
  · intro j hj d hd
    simp only [physical_read]
    rw [results j hj d hd]
    have launch : active c.observation := by
      rcases hj with ⟨hb,hh⟩
      exact ⟨lt_of_le_of_lt (Nat.zero_le _) hb,lt_of_le_of_lt (Nat.zero_le _) hh⟩
    exact real_row (host c.observation) _ (checked_admissible _ h.metadata _
      (checked_content _ _ _ _ (h.contents launch) _ (real_decoding (view c m)))) logics values finite j hj d hd

#axiomsClean tensor_correct
#specNonCircular realTarget avoiding [invokeReal,Host.wrapper,FPValue.source,FPValue.outputValues]
end
end VeriTile.Bench.Repairs.SoftmaxReduceV.Correct
