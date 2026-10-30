import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceLoop

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceBounds
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem stop_common (abi : SourceABI) (p : KernelArgs) :
    (SourceParts.loopParts abi p).stop = (SourceParts.loopParts primaryHalf p).stop := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem eval_start {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α) :
    evalOp M none (SourceParts.loopParts abi p).start s = some (fun _ => 0) := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem eval_stride {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α) :
    evalOp M none (SourceParts.loopParts abi p).stride s = some (fun _ => 1) := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem eval_stop {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α)
    (hk : s.regs .int [] "K" = some (fun _ => (p.inner : Int))) :
    evalOp M none (SourceParts.loopParts abi p).stop s =
      some (fun _ => (((p.inner+15)/16 : Nat) : Int)) := by
  rw [stop_common]
  have hp : ¬ (p.inner : Int)+16 < 1 := by omega
  simp [SourceParts.loopParts,SourceParts.active,SourceParts.afterTile,SourceParts.afterLength,
    SourceParts.thenBody,SourceParts.outer,Source.intRange?,kernel,primaryHalf,sourceFace,
    sourceHalfSingle323232,ComputeKernel.surfaceBody,evalOp_unfold,Option.bind,
    numeric,bop,numericLt,integralFloorDiv,integralMod,hk,hp,SourceBody.signed_block_count]

theorem loop_step {α : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (c : SourceBody.Coordinates) (s t : State α)
    (initial : Values α .real [32,32]) (h : SourceBody.Ready p c s t initial) :
    ∃ final, step (SourceBody.install M dot)
      (.forRangeInt "k" (SourceParts.loopParts abi p).start (SourceParts.loopParts abi p).stop
        (SourceParts.loopParts abi p).stride (SourceParts.body abi p)) t = some final ∧
      SourceBody.Ready p c s final (SourceBody.accumulated M dot abi p c s 0 initial) := by
  obtain ⟨final,hr,hf⟩ := SourceBody.loop_run M dot abi p c s t 0 initial h
  refine ⟨final,?_,hf⟩
  simpa only [step,eval_start,eval_stop _ _ _ _ h.inner,eval_stride,Option.bind_some] using hr

#axiomsClean eval_start
#axiomsClean eval_stride
#axiomsClean eval_stop
#axiomsClean loop_step
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceBounds
