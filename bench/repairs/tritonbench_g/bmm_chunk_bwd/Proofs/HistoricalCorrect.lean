import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Real
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.BmmChunkBwd.HistoricalCorrect
open VeriTile.Triton
noncomputable section

structure Result where
  object : Nat
  pointer : Nat
  shape : List Nat
  device : VeriTile.Torch.Device
  dtype : VeriTile.Torch.DType
  state : BlockState

def result (p : Metadata) (a : Direct.Allocation p) (s : BlockState) : Result :=
  ⟨a.object,a.pointer,(p.tensor .a).shape,(p.tensor .a).device,.floating .fp32,s⟩

theorem python_checks_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p := checkArguments_iff p

/-- Full returned gradient over arbitrary Real inputs for every finite tuner
trace, with direct supplied-output identity or a fresh ordinary allocation. -/
specification bmm_chunk_bwd_real (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (inputs : Mathematical.Inputs) (trials : List (Fin 9)) (chosen : Fin 9) :
    ∃ out : Result,checkInputs p=.ok () ∧ Direct.executeReal p a inputs trials chosen=some out.state ∧
      out.object=a.object ∧ out.pointer=a.pointer ∧ out.shape=(p.tensor .a).shape ∧
      out.device=(p.tensor .a).device ∧ out.dtype=.floating .fp32 ∧
      (p.hasOutput=Bool.true → out.object=a.objects .output ∧ out.pointer=base p .output) ∧
      (p.hasOutput=Bool.false → ∀ r,Used p r → out.object≠a.objects r) ∧
      (∀ b s g n,b<batch p → s<seqlen p → g<groups p → n<width p →
        out.state.readMem (Direct.flat p a).flat ((Direct.flat p a).addr (call p).Out (Direct.tensorIndex p b s g n))=
          Mathematical.product p inputs b s g n) ∧
      (∀ r off,¬Direct.OutputCell p a r off →
        out.state.mem r off=((Direct.flat p a).flattenState (Direct.realState p inputs)).mem r off) := by
  obtain ⟨s,run,values,frame⟩ := Direct.real_complete p h a inputs trials chosen
  refine ⟨result p a s,(checkInputs_iff p).mpr h,run,rfl,rfl,rfl,rfl,rfl,?_,?_,values,frame⟩
  · intro supplied; simp [result,Direct.Allocation.object,Direct.Allocation.pointer,supplied]
  · intro absent r used
    simpa [result,Direct.Allocation.object,absent] using a.freshIdentity absent r used

#axiomsClean python_checks_iff
#axiomsClean bmm_chunk_bwd_real
#specNonCircular Mathematical.product avoiding [Direct.executeReal,kernel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.HistoricalCorrect
