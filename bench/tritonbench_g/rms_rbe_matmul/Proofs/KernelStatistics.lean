import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelPrefix
import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelTailModel

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure TailReady (p : Parameters) (s t : State α) (values : TileIndex [16,64] → α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  batch : t.regs .nat [] "pid_batch"=some (fun _ => s.pids 0)
  tileM : t.regs .nat [] "pid_m"=some (fun _ => pidM p s)
  tileN : t.regs .nat [] "pid_n"=some (fun _ => pidN p s)
  product : t.regs .real [16,64] "accumulator"=some values

theorem statistics_run (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (cur : Nat) (acc : Accumulator α) (h : Ready p s t cur acc) :
    ∃ u, run (engine a) (statistics fp8 p) t=some u ∧
      TailReady p s u (normalized a p acc) := by
  cases fp8
  all_goals
    simp [statistics,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      Region.cast,ofFloat,toFloat,State.setReg,
      engine,Algebra.withDefaultPrecision,resolvePrecision,h.product,h.squares]
  all_goals constructor
  all_goals simp [State.setReg,h.mem,h.pids,h.undef,h.batch,h.tileM,h.tileN,
    normalized,norm,bop]
  all_goals first | rfl | (funext i; simp [normalized,norm,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,one_div])
  all_goals rcases i with ⟨i,j,⟨⟩⟩; rfl

#axiomsClean statistics_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
