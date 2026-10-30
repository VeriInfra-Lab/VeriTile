import bench.tritonbench_g.block_sparse_attn.VisitPrefix
import bench.tritonbench_g.block_sparse_attn.ChannelLoop
import bench.tritonbench_g.block_sparse_attn.VisitSuffix

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem loopBody_partition (native : Bool) (p : Parameters) :
    loopBody native p = (body p).take 3 ++
      [.forLoop "score_block" p.blocks (channelBody native p)] ++ suffix native p := by
  cases native <;> rfl

/-- One complete CSR occurrence, including its real metadata load, all
channel blocks, causal normalization, value load, and accumulator update. -/
theorem visitExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (slot : Int) (h : Runtime a p initial carry current)
    (integers : Int32Memory initial) :
    ∃ final, run a.model (loopBody native p)
      (current.setReg "col_idx_idx" .int [] (fun _ => slot))=some final ∧
      Runtime a p initial (update a native p initial (csrColumn p initial slot) carry) final := by
  let start := visitPrefixExecution a p initial current carry slot h integers
  have hi : ChannelInvariant a p initial carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) 0) start.val :=
    visitPrefixInvariant a p initial current carry slot h integers
  obtain ⟨mid,hm,hmi⟩ := channelLoopExecution a native p initial start.val carry
    (csrColumn p initial slot) 0 (by omega) hi
  let finish := visitSuffixExecution a native p initial mid carry (csrColumn p initial slot) hmi
  have hloop : run a.model [.forLoop "score_block" p.blocks (channelBody native p)] start.val=some mid := by
    simp [run,FP.Structural.step,hm]
  refine ⟨finish.val,?_,visitSuffixInvariant a native p initial mid carry (csrColumn p initial slot) hmi⟩
  rw [loopBody_partition,run_append,run_append,start.property]
  simp only [Option.bind_some,hloop,finish.property]

#axiomsClean loopBody_partition
#axiomsClean visitExecution
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
