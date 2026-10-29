import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.KernelInterface
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.ABI

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.KernelNativeResults
open VeriTile.Triton FP.Structural
noncomputable section

/-- Raw FP32 words and signed integer metadata from the same initial memory. -/
def input (m : Host.Memory NativeGraph.Word) : Host.Input NativeGraph.Word :=
  Host.decodeMemory (fun w => w.toInt.toNat) m

def nativeEvaluate (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (m : Host.Memory NativeGraph.Word) (j : Host.Job) :=
  NativeGraph.evaluate (NativeGraph.profilePlan (profiles j)) ops c (input m) j

def nativeTarget (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (d : Nat) :=
  NativeGraph.row (NativeGraph.profilePlan (profiles j)) ops c (input m) j d

theorem native_total (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (s : Nat → NativeGraph.Word) :
    ∃ t,nativeEvaluate profiles ops c m j s=some t := by
  obtain ⟨u,run,post⟩ := FPValue.execution_memory (NativeGraph.algebra (NativeGraph.profilePlan (profiles j)))
    (Host.parameters c)
    (Host.launch c (NativeGraph.nodeInput (input m)) (fun a => NativeGraph.atom (s a)) j)
    (by change 0<64;decide)
  exact ⟨NativeGraph.observe (NativeGraph.profilePlan (profiles j)) ops c j u,
    by simp [nativeEvaluate,NativeGraph.evaluate,run]⟩

/-- Complete source grid and independent ordered native outputs for all raw
FP32 input words, with the exact exterior-memory frame. The selected profile
retains the original reductions, replicas, floor clamp, FMA and final division. -/
theorem kernel_native (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (m : Host.Memory NativeGraph.Word)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) :
    ∃ final,RawKernel.runGrid c (nativeEvaluate profiles ops c m) work m=.ok () final ∧
      (∀ j,Host.Legal c j → ∀ d,d<c.dimensions →
        final "Out" (Host.address c j d)=nativeTarget profiles ops c m j d) ∧
      (∀ r i,Host.Outside c 0 r i → final r i=m r i) := by
  exact Checked.source_whole c (nativeEvaluate profiles ops c m) (nativeTarget profiles ops c m)
    ⟨fun j => NativeGraph.source_law (NativeGraph.profilePlan (profiles j)) ops c (input m) j⟩
    (native_total profiles ops c m) work complete m

theorem local_refinement (profiles : Host.Job → NativeGraph.Profile)
    (actual reference : NativeGraph.Instructions) (ports : NativeGraph.InstructionAgreement actual reference)
    (c : Host.Call) (m : Host.Memory NativeGraph.Word) (j : Host.Job) (d : Nat) :
    nativeTarget profiles actual c m j d=nativeTarget profiles reference c m j d :=
  NativeGraph.row_agreement (NativeGraph.profilePlan (profiles j)) actual reference ports c (input m) j d

#axiomsClean native_total
#axiomsClean kernel_native
#axiomsClean local_refinement
#specNonCircular nativeTarget avoiding [nativeEvaluate,RawKernel.runGrid,FPValue.source,FPValue.sourceBody,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.KernelNativeResults
