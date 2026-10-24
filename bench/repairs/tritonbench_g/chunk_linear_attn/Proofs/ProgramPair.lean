import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.PairIO
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOFrame
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOLayoutFacts

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.ProgramPair
open VeriTile Triton FP.Structural
noncomputable section

/-- Task-local view of the existing finite reduction-stage language. IO,
encoders, scalar ABI binding and output observers are fixed in the type.
Execution and safety are the shared actual stage runner and trace judgment;
there is no execution callback or mathematical result field. -/
structure Program (io : ReductionProgramIO (BitVec 32) NativeTerms.Term) where
  stages : List (ReductionStage NativeTerms.Term)

/-- Both actual complete executions are safe and successful; all nine observed
channels agree, all memory cells agree, and each exact exterior frame holds.
The input domain is the original unrestricted raw-word channel domain. -/
def Equivalent (io : ReductionProgramIO (BitVec 32) NativeTerms.Term)
    (lhs rhs : Program io) : Prop :=
  ∀ vals before, io.InputBinding vals before →
    (∀ bounds, io.InBounds bounds vals →
      ReductionScheduleSafe bounds lhs.stages before ∧ ReductionScheduleSafe bounds rhs.stages before) ∧
    ∃ a b, runReductionStages lhs.stages before=some a ∧ runReductionStages rhs.stages before=some b ∧
      (∀ o j, io.outputMask o vals j →
        mapCell (io.observe o j) (a.mem io.outputs[o].buf (io.write o vals j)) =
          mapCell (io.observe o j) (b.mem io.outputs[o].buf (io.write o vals j))) ∧
      a.mem=b.mem ∧ io.Frame vals before a ∧ io.Frame vals before b

instance (io : ReductionProgramIO (BitVec 32) NativeTerms.Term) : Spec.ProgramSyntax (Program io) where
  Statement := ReductionStage NativeTerms.Term
  Signature := MemoryIO (BitVec 32)
  signature := fun _ => io.toMemoryIO
  body := Program.stages
  structural := some (Equivalent io)

def original (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config) (scale : BitVec 32)
    (work : KernelSchedule.Work) : Program (IO.native profile c scale work) :=
  ⟨IO.stages NativeTerms.algebra NativeTerms.dotPrimitive (KernelExecution.symbolicCall c.call) work⟩

def hoisted (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config) (scale : BitVec 32)
    (work : KernelSchedule.Work) : Program (IO.native profile c scale work) :=
  ⟨PairIO.stages NativeTerms.algebra NativeTerms.dotPrimitive (KernelExecution.symbolicCall c.call) work⟩

theorem equivalent (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config) (scale : BitVec 32)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work) :
    Equivalent (IO.native profile c scale work) (original profile c scale work) (hoisted profile c scale work) := by
  intro vals before _
  have symbolicLegal : KernelSchedule.Legal (KernelExecution.symbolicCall c.call) work :=
    ⟨legal.forwardState,legal.forwardOutput,legal.backwardState,legal.backwardGradient⟩
  refine ⟨?_,?_⟩
  · intro bounds backed
    have realBounds := IOLayoutFacts.native_bounds c vals bounds backed
    exact ⟨IOSafety.stages NativeTerms.algebra NativeTerms.dotPrimitive c vals work legal 1 bounds realBounds before,
      PairIO.stages_safe NativeTerms.algebra NativeTerms.dotPrimitive c vals work legal 1 bounds realBounds before⟩
  · obtain ⟨a,ar,am⟩ := IORun.stages NativeTerms.algebra NativeTerms.dotPrimitive
      (KernelExecution.symbolicCall c.call) work symbolicLegal before
    obtain ⟨b,br,bm⟩ := PairIO.stages_run NativeTerms.algebra NativeTerms.dotPrimitive
      (KernelExecution.symbolicCall c.call) work symbolicLegal before
    have same : a.mem=b.mem := am.trans bm.symm
    have frame : (IO.native profile c scale work).Frame vals before a := by
      intro r offset outside
      rw [am]
      exact KernelInterface.complete_frame NativeTerms.algebra NativeTerms.dotPrimitive
        (KernelExecution.symbolicCall c.call) before.mem r offset
        (IOFrame.outside c vals r offset (IOLayoutFacts.native_outside c vals r offset outside))
    refine ⟨a,b,ar,br,?_,same,frame,?_⟩
    · intro o j _
      rw [same]
    · intro r offset outside
      rw [←same]
      exact frame r offset outside

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.ProgramPair
