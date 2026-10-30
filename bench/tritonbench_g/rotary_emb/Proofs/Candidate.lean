import bench.tritonbench_g.rotary_emb.Proofs.TypedExecution

namespace VeriTile.Bench.Repairs.RotaryEmb.Candidate
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Faithful transport of the historical candidate: commute only the integer
head-block multiplication. Every subsequent source statement is retained. -/
def optimizedPrefix (p : Kernels.Layout) : ComputeKernel := triton {
  cur_head_index = tl.program_id(0).to(tl.int64)
  cur_seq_index = tl.program_id(1).to(tl.int64)
  cur_head_range = $((p.BLOCK_HEAD : Int)) * cur_head_index +
    tl.arange(0, $(p.BLOCK_HEAD)).to(tl.int64)
}

/-- Compatibility name for the complete candidate source in Kernels.lean. -/
abbrev optimized := Kernels.indexCommuted

theorem prefix_equiv {A : Type} [Inhabited A] (M : Algebra A)
    (p : Kernels.Layout) (s : State A) :
    run M ((Kernels.kernel p).surfaceBody.take 3) s=run M (optimizedPrefix p).surfaceBody s := by
  simp [Kernels.kernel,optimizedPrefix,ComputeKernel.surfaceBody,List.take,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,bop,numeric,
    ComputeDType.eraseDType,Int.mul_comm]
  congr 1
  funext i
  simp [bop,Int.mul_comm]

/-- Full structural equivalence, including all registers and every memory
cell, for arbitrary payload semantics and arbitrary initial aliases. -/
theorem execution_equiv {A : Type} [Inhabited A] (M : Algebra A)
    (p : Kernels.Layout) (s : State A) :
    exec M (Kernels.kernel p) s=exec M (optimized p) s := by
  change run M (Kernels.kernel p).surfaceBody s=
    run M ((optimizedPrefix p).surfaceBody++(Kernels.kernel p).surfaceBody.drop 3) s
  conv_lhs => rw [←List.take_append_drop 3 (Kernels.kernel p).surfaceBody]
  rw [run_append,prefix_equiv,run_append]

#axiomsClean execution_equiv
end
end VeriTile.Bench.Repairs.RotaryEmb.Candidate
