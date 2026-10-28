import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Correct
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Observations
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.NativePublic
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Contents

/-! Complete checked Tensor applications. Original identities, physical words,
little-endian bytes, exterior frames and exceptional-value domains are retained. -/
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.TensorBridge
open VeriTile.Triton FP.Structural Checked
noncomputable section

def realEvaluate (c : TensorCall) (m : Host.Memory (WithBot ℝ)) :=
  Host.evaluate RealModel.model (host c.observation) (Host.decodeMemory Host.realIndex (view c m))
def invokeReal (c : TensorCall) (work : List Host.Job) (m : Host.Memory (WithBot ℝ)) :=
  if checkInput c.observation (realObserved (view c m) "Starts")
      (realObserved (view c m) "Lengths") (realObserved (view c m) "Locations") = .ok () then
    invoke c work (realEvaluate c m) m else .error "input contract rejected" m

def realTarget (c : TensorCall) (m : Host.Memory (WithBot ℝ)) (logics : Nat → WithBot ℝ) (values : Nat → ℝ)
    (j : Host.Job) (d : Nat) :=
  mathematical (host c.observation) (Host.decodeMemory Host.realIndex (view c m)) logics values j d

theorem tensor_correct (c : TensorCall)
    (m : Host.Memory (WithBot ℝ))
    (h : InputContract c.observation (realObserved (view c m) "Starts")
      (realObserved (view c m) "Lengths") (realObserved (view c m) "Locations"))
    (metadata : Host.RealMetadataEncoding (host c.observation) (view c m))
    (logics : Nat → WithBot ℝ) (values : Nat → ℝ)
    (finite : NormalizableInputs (host c.observation) (Host.decodeMemory Host.realIndex (view c m)) logics values)
    (work : List Host.Job) (complete : work.Perm (Host.workset (host c.observation))) :
    ∃ result,invokeReal c work m=.ok result result.memory ∧ result.returned=none ∧
      result.output=c.output.tensor ∧
      (∀ j,Host.Legal (host c.observation) j → ∀ d,d < dimensions c.observation →
        tensorValue result.output result.memory (Host.address (host c.observation) j d)=
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
    simp only [tensorValue_object,physical_read]
    rw [results j hj d hd]
    have launch : active c.observation := by
      rcases hj with ⟨hb,hh⟩
      exact ⟨lt_of_le_of_lt (Nat.zero_le _) hb,lt_of_le_of_lt (Nat.zero_le _) hh⟩
    exact real_row (host c.observation) _ (checked_admissible _ h.metadata _
      (checked_content _ _ _ _ (h.contents launch) _ (real_decoding (view c m)))) logics values finite j hj d hd

#axiomsClean tensor_correct
#specNonCircular realTarget avoiding [invokeReal,Host.wrapper,FPValue.source,FPValue.outputValues,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.TensorBridge

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.TensorBridge
open VeriTile.Triton FP.Structural Checked
noncomputable section

def nativeEvaluate (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : TensorCall) (m : Host.Memory NativeGraph.Word) (j : Host.Job) :=
  NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops (host c.observation) (signedNativeInput (view c m)) j

def invokeNative (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : TensorCall) (work : List Host.Job) (m : Host.Memory NativeGraph.Word) :=
  if checkInput c.observation (nativeObserved (view c m) "Starts")
      (nativeObserved (view c m) "Lengths") (nativeObserved (view c m) "Locations") = .ok () then
    invoke c work (nativeEvaluate profiles ops c m) m else .error "input contract rejected" m

def nativeTarget (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : TensorCall) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (d : Nat) :=
  NativeGraph.row (NativeGraph.profilePlan (profiles j)) ops (host c.observation) (signedNativeInput (view c m)) j d

theorem native_total (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : TensorCall) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (s : Nat → NativeGraph.Word) :
    ∃ t,nativeEvaluate profiles ops c m j s=some t := by
  obtain ⟨u,run,post⟩ := FPValue.execution_memory (NativeGraph.algebra (NativeGraph.profilePlan (profiles j)))
    (Host.parameters (host c.observation))
    (Host.launch (host c.observation) (NativeGraph.nodeInput (signedNativeInput (view c m)))
      (fun a => NativeGraph.atom (s a)) j) (by change 0 < 64;decide)
  exact ⟨NativeGraph.observe (NativeGraph.profilePlan (profiles j)) ops (host c.observation) j u,
    by simp [nativeEvaluate,NativeGraph.evaluate,run]⟩

theorem tensor_fp32 (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : TensorCall) (m : Host.Memory NativeGraph.Word)
    (h : InputContract c.observation (nativeObserved (view c m) "Starts")
      (nativeObserved (view c m) "Lengths") (nativeObserved (view c m) "Locations"))
    (work : List Host.Job) (complete : work.Perm (Host.workset (host c.observation))) :
    ∃ result,invokeNative profiles ops c work m=.ok result result.memory ∧ result.returned=none ∧
      result.output=c.output.tensor ∧
      (∀ j,Host.Legal (host c.observation) j → ∀ d,d < dimensions c.observation →
        tensorValue result.output result.memory (Host.address (host c.observation) j d)=
          nativeTarget profiles ops c m j d) ∧
      (∀ r i,Exterior c r i → result.memory r i=m r i) ∧
      (∀ r b,Exterior c r (b/4) → memoryByte result.memory r b=memoryByte m r b) := by
  let source : Host.SourceLawGeneral (host c.observation) (nativeEvaluate profiles ops c m)
      (nativeTarget profiles ops c m) :=
    ⟨fun j => NativeGraph.source_law (NativeGraph.profilePlan (profiles j)) ops _ _ j⟩
  obtain ⟨final,run,results,frame⟩ := source_whole (host c.observation) (nativeEvaluate profiles ops c m) _ source
    (native_total profiles ops c m) work complete (view c m)
  refine ⟨⟨c.output.tensor,none,physical c m final⟩,?_,rfl,rfl,?_,physical_frame c m final frame,byte_frame c m _ (physical_frame c m final frame)⟩
  · simp [invokeNative,(check_input_iff _ _ _ _).mpr h,invoke,run,liftResult]
  · intro j hj d hd
    simpa only [tensorValue_object,physical_read] using results j hj d hd

theorem local_refinement (profiles : Host.Job → NativeGraph.Profile)
    (actual reference : NativeGraph.Instructions) (ports : NativeGraph.InstructionAgreement actual reference)
    (c : TensorCall) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (d : Nat) :
    nativeTarget profiles actual c m j d=nativeTarget profiles reference c m j d :=
  NativeGraph.row_agreement (NativeGraph.profilePlan (profiles j)) actual reference ports _ _ _ _

#axiomsClean tensor_fp32
#axiomsClean local_refinement
#specNonCircular nativeTarget avoiding [invokeNative,Host.wrapper,FPValue.source,FPValue.sourceBody,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.TensorBridge

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.TensorBridge
open VeriTile.Triton FP.Structural Checked
noncomputable section

/-- Constant gathered values remain that constant even after arbitrarily many
all-negative-infinity blocks, provided a finite active logit eventually occurs.
In particular this proves the independent result one for the saved defect. -/
theorem tensor_constant (c : TensorCall) (m : Host.Memory (WithBot ℝ))
    (h : InputContract c.observation (realObserved (view c m) "Starts")
      (realObserved (view c m) "Lengths") (realObserved (view c m) "Locations"))
    (metadata : Host.RealMetadataEncoding (host c.observation) (view c m))
    (logics : Nat → WithBot ℝ) (value : ℝ)
    (scope : NormalizableInputs (host c.observation)
      (Host.decodeMemory Host.realIndex (view c m)) logics (fun _ => value))
    (work : List Host.Job) (complete : work.Perm (Host.workset (host c.observation))) :
    ∃ result,invokeReal c work m=.ok result result.memory ∧ result.returned=none ∧
      result.output=c.output.tensor ∧
      (∀ j,Host.Legal (host c.observation) j → ∀ d,d < dimensions c.observation →
        tensorValue result.output result.memory (Host.address (host c.observation) j d)=↑value) ∧
      (∀ r i,Exterior c r i → result.memory r i=m r i) := by
  obtain ⟨result,run,returned,output,values,frame⟩ :=
    tensor_correct c m h metadata logics (fun _ => value) scope work complete
  refine ⟨result,run,returned,output,?_,frame⟩
  intro j hj d hd
  rw [values j hj d hd]
  congr 1
  exact RealModel.target_constant _ _ value (scope.positive j hj)

#axiomsClean tensor_constant
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.TensorBridge
