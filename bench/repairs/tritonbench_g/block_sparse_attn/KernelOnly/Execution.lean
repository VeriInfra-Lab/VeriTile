import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Loop
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Output

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def execution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial : State (Replicas α)) :=
  run a.model (if native then selectedSource p else (source p).surfaceBody)
    (bindScale (broadcast (a.argument p.scale)) initial)

theorem execution_partition (native : Bool) (p : Parameters) :
    (if native then selectedSource p else (source p).surfaceBody) = prelude p ++
      [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
        (.ref .int [] "end_l") (.constNat 1) (loopBody native p)] ++ epilogue p := by
  cases native
  · exact partition p
  · rfl

/-- Successful execution of the complete shared source (or its selected
local native lowering), followed by the actual masked store. The conclusion
is the independent initial-memory target; no graph binding is a premise. -/
theorem complete {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial : State (Replicas α)) (integers : Int32Memory initial) :
    ∃ final, execution a native p initial=some final ∧
      final.mem=outputMemory p initial (output a native p initial) := by
  let start := prefixExecution a.model p (bindScale (broadcast (a.argument p.scale)) initial) integers
  have hi : Runtime a p initial (accumulation a native p initial 0) start.val :=
    initialRuntime a p initial integers
  obtain ⟨mid,hm,hmi⟩ := outerLoopExecution a native p initial start.val 0 (by omega) hi integers
  let finish := epilogueExecution a p initial mid
    (accumulation a native p initial (visitCount p initial)) hmi
  have loopRun : run a.model
      [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
        (.ref .int [] "end_l") (.constNat 1) (loopBody native p)] start.val=some mid := by
    simpa [run,FP.Structural.step,evalOp_unfold,hi.first,hi.last] using hm
  refine ⟨finish.val,?_,epilogueMemory a p initial mid _ hmi⟩
  unfold execution
  rw [execution_partition,run_append,run_append,start.property]
  simp only [Option.bind_some,loopRun,finish.property]

#axiomsClean execution_partition
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
