import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelModel

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def xPointer (p : Parameters) (s : State α) (step : Nat) : Values α .ptr [16,64] :=
  fun i => ("X", xOffset p s step i)
def wPointer (p : Parameters) (s : State α) (step : Nat) : Values α .ptr [64,64] :=
  fun i => ("W", wOffset p s step i)
def rPointer (step : Nat) : Values α .ptr [1,64] := fun i => ("RMS", rOffset step i)
structure Ready (p : Parameters) (s t : State α) (step : Nat) (acc : Accumulator α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  batch : t.regs .nat [] "pid_batch"=some (fun _ => s.pids 0)
  tileM : t.regs .nat [] "pid_m"=some (fun _ => pidM p s)
  tileN : t.regs .nat [] "pid_n"=some (fun _ => pidN p s)
  offk : t.regs .nat [64] "offs_k"=some (fun i => i.1.val)
  xptr : t.regs .ptr [16,64] "x_ptrs"=some (xPointer p s step)
  wptr : t.regs .ptr [64,64] "w_ptrs"=some (wPointer p s step)
  rptr : t.regs .ptr [1,64] "rms_w_ptrs"=some (rPointer step)
  product : t.regs .real [16,64] "accumulator"=some acc.product
  squares : t.regs .real [16,64] "x_sum"=some acc.squares

theorem prefix_run (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) :
    ∃ t, run (engine a) (prelude fp8 p) s=some t ∧ Ready p s t 0 (initial a) := by
  cases fp8 <;>
    simp [prelude,kernel,TypedSource.kernel,TypedSource.source,arguments,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals constructor
  all_goals simp [State.setReg,pidM,pidN,xPointer,wPointer,rPointer,xOffset,wOffset,rOffset,
    row,col,initial,zero]
  all_goals first | rfl | (funext i; simp [bop,initial,zero,xPointer,wPointer,rPointer,
    xOffset,wOffset,rOffset,row,col,pidM,pidN])

#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
