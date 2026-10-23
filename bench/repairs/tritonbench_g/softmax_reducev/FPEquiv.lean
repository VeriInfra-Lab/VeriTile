import bench.repairs.tritonbench_g.softmax_reducev.Proofs.NativePublic
import bench.repairs.tritonbench_g.softmax_reducev.Proofs.Contents

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.FPEquiv
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
        result.memory c.output.region (c.output.tensor.storageOffset+Host.address (host c.observation) j d)=
          nativeTarget profiles ops c m j d) ∧
      (∀ r i,Exterior c r i → result.memory r i=m r i) := by
  let source : Host.SourceLawGeneral (host c.observation) (nativeEvaluate profiles ops c m)
      (nativeTarget profiles ops c m) :=
    ⟨fun j => NativeGraph.source_law (NativeGraph.profilePlan (profiles j)) ops _ _ j⟩
  obtain ⟨final,run,results,frame⟩ := source_whole (host c.observation) (nativeEvaluate profiles ops c m) _ source
    (native_total profiles ops c m) work complete (view c m)
  refine ⟨⟨c.output.tensor,none,physical c m final⟩,?_,rfl,rfl,?_,physical_frame c m final frame⟩
  · simp [invokeNative,(check_input_iff _ _ _ _).mpr h,invoke,run,liftResult]
  · intro j hj d hd
    simpa only [physical_read] using results j hj d hd

theorem local_refinement (profiles : Host.Job → NativeGraph.Profile)
    (actual reference : NativeGraph.Instructions) (ports : NativeGraph.InstructionAgreement actual reference)
    (c : TensorCall) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (d : Nat) :
    nativeTarget profiles actual c m j d=nativeTarget profiles reference c m j d :=
  NativeGraph.row_agreement (NativeGraph.profilePlan (profiles j)) actual reference ports _ _ _ _

#axiomsClean tensor_fp32
#axiomsClean local_refinement
#specNonCircular nativeTarget avoiding [invokeNative,Host.wrapper,FPValue.source]
end
end VeriTile.Bench.Repairs.SoftmaxReduceV.FPEquiv
