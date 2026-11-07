import bench.tritonbench_g.parallel_attention.Proofs.Support.QueryWhole
import bench.tritonbench_g.parallel_attention.Proofs.Support.FPBackwardWhole
import bench.tritonbench_g.parallel_attention.Proofs.Support.ShellSyntax

/-! Proof views into the complete pasted caller. These excerpts are used only
for execution reasoning; Kernels.lean remains the source authority. -/
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
open VeriTile.Triton FP.Structural
noncomputable section

def kernel (c : QueryConfig) : ComputeKernel :=
  shellSource c.q c.k c.v c.do_ c.dz c.dq c.dk c.dv c.i_bh c.i_c c.i_k c.i_v c.i_h
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

structure Registers (c : BackwardConfig) (s : State α) : Prop where
  head : s.regs .nat [] "i_bh"=some (fun _=>c.i_bh)
  chunk : s.regs .nat [] "i_c"=some (fun _=>c.i_c)
  keyBlock : s.regs .nat [] "i_k"=some (fun _=>c.i_k)
  valueBlock : s.regs .nat [] "i_v"=some (fun _=>c.i_v)

def config (c : QueryConfig) (s : State α) : QueryConfig :=
  {c with
    i_bh := s.pids 2
    i_c := s.pids 1
    i_k := s.pids 0 / ((c.V+c.BV-1)/c.BV)
    i_v := s.pids 0 % ((c.V+c.BV-1)/c.BV)
    i_h := s.pids 2 % c.H}

def callerPrefix (c : QueryConfig) : List ComputeStmt := (kernel c).surfaceBody.take 5

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (s : State α) :
    ∃ t,run M (callerPrefix c) s=some t ∧ Registers (config c s).toBackwardConfig t ∧
      t.mem=s.mem ∧ t.pids=s.pids ∧ t.undef=s.undef := by
  simp [callerPrefix,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop]
  constructor <;> simp [config,integralFloorDiv,integralMod,bop]
  all_goals rfl

/-- The caller's index arguments are supplied by its own PID decoder. -/
theorem kernel_config (c : QueryConfig) (s : State α) : kernel (config c s)=kernel c:=rfl

#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
