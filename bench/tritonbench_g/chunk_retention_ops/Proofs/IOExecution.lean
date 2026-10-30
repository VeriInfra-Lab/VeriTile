import bench.tritonbench_g.chunk_retention_ops.IO

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelPacket
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (foldRange_zero foldRange_succ scatter)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000

namespace H
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
  (accumulated advance stored finalStored written initValue update)
end H

private theorem forward_accumulated (M : Algebra α) (D : DotPrimitive α) (a : Call)
    (s : State α) (n : Nat) :
    let start := (launch (engine (withDot M D)) a).initial s
    let acc := foldRange (H.advance M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) start)
      0 n 1 ⟨H.initValue M (KernelOutput.shape a.shape) a.inputState start,start⟩
    acc.1=KernelForward.values M D a.shape a.pid s.mem a.inputState n ∧
    acc.2.mem=foldRange (fun n mem=>KernelForward.writeHistory M a.shape a.pid n
      (KernelForward.values M D a.shape a.pid s.mem a.inputState n) mem) 0 n 1 s.mem := by
  induction n with
  | zero =>
    dsimp only
    simp only [foldRange_zero]
    exact ⟨rfl,rfl⟩
  | succ n ih =>
    dsimp only at ih ⊢
    rw [foldRange_succ,foldRange_succ]
    dsimp only [H.advance]
    constructor
    · rw [ih.1]
      rfl
    · simp only [H.stored,masked_scatter_memory]
      rw [ih.1,ih.2]
      rfl

private theorem forward_written (M : Algebra α) (D : DotPrimitive α) (a : Call) (s : State α) :
    (H.written M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) a.inputState a.outputState
      ((launch (engine (withDot M D)) a).initial s)).mem=
      KernelForward.memory M D a.shape a.pid s.mem s.mem a.inputState a.outputState := by
  have h := forward_accumulated M D a s a.shape.chunks
  dsimp only at h
  change (H.accumulated M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) a.inputState
    ((launch (engine (withDot M D)) a).initial s)).1=_ ∧
    (H.accumulated M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) a.inputState
    ((launch (engine (withDot M D)) a).initial s)).2.mem=_ at h
  unfold H.written KernelForward.memory KernelForward.writeFinal
  cases hf : a.outputState
  · exact h.2
  · simp only [Bool.true_eq,if_true,H.finalStored,masked_scatter_memory]
    rw [h.1,h.2]
    rfl

private theorem output_written (M : Algebra α) (D : DotPrimitive α) (a : Call) (s : State α) :
    (KernelOutput.S.written M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) a.scale
      ((launch (engine (withDot M D)) a).initial s)).mem=
      KernelOutput.memory M D a.shape a.scale a.pid s.mem s.mem := by
  simp only [KernelOutput.S.written,masked_scatter_memory]
  rfl

/-- Reuse the original complete source proofs with arbitrary pre-existing
registers and undefined-value functions. Their actual loads and updates depend
only on the declared memory and installed program coordinates. -/
theorem schedule_result (M : Algebra α) (D : DotPrimitive α) (a : Call) (s : State α) :
    ∃ final,runSchedule [launch (engine (withDot M D)) a] s=some final ∧
      final.mem=KernelPacket.reference M D a s.mem := by
  rw [runSchedule_singleton]
  rcases a with ⟨c,stage,scale,seed,final,pid⟩
  cases stage with
  | forwardState =>
    obtain ⟨t,ran,result⟩ := KernelForward.program M (KernelOutput.sharedDot D) c seed final
      ((launch (engine (withDot M D)) ⟨c,.forwardState,scale,seed,final,pid⟩).initial s)
    exact ⟨t,ran,result.trans (forward_written M D ⟨c,.forwardState,scale,seed,final,pid⟩ s)⟩
  | forwardOutput =>
    obtain ⟨t,ran,result⟩ := KernelOutput.program M (KernelOutput.sharedDot D) c scale
      ((launch (engine (withDot M D)) ⟨c,.forwardOutput,scale,seed,final,pid⟩).initial s)
    exact ⟨t,ran,result.trans (output_written M D ⟨c,.forwardOutput,scale,seed,final,pid⟩ s)⟩
  | backwardState =>
    obtain ⟨t,ran,result,_⟩ := KernelBackward.program M D (parameters c) scale seed final s.mem
      ((launch (engine (withDot M D)) ⟨c,.backwardState,scale,seed,final,pid⟩).initial s)
      (KernelBackward.InputEq.refl _) (by simp [parameters])
    exact ⟨t,ran,result⟩
  | backwardGradient =>
    obtain ⟨t,ran,result⟩ := KernelGradient.gradientProgram M D (parameters c) scale
      ((launch (engine (withDot M D)) ⟨c,.backwardGradient,scale,seed,final,pid⟩).initial s)
      (by simp [parameters])
    exact ⟨t,ran,result⟩

#axiomsClean schedule_result
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
