import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.FusedContext

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
open VeriTile Triton FP.Structural Kernel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (ForwardConfig BackwardConfig DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def forward (c : ForwardConfig) : ComputeKernel :=
  FPStages.fusedForwardSource c.q c.k c.v c.o c.s_qk_h c.s_qk_t c.s_qk_d
    c.s_vo_h c.s_vo_t c.s_vo_d c.scale c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def query (c : BackwardConfig) (dq : RegionName) : ComputeKernel := FPQuery.fusedSource c dq

def loopBody (k : ComputeKernel) (n : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := k.surfaceBody
  match code[n]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ b)=>b
  | some (ComputeStmt.forRange _ _ _ _ b)=>b
  | _=>[]

theorem overwrite (s : State α) (n : RegName) (d : TileDType) (sh : TileShape)
    (u v : Values α d sh) : (s.setReg n d sh u).setReg n d sh v=s.setReg n d sh v := by
  simp only [State.setReg]
  congr 1
  funext d' sh' n'
  split <;> simp_all

theorem forward_local_run (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig) (s : State α) :
    run (withDot M D) (((loopBody (FPStages.forwardKernel c) 16).drop 3).take 2) s=
      run (withDot M D) (((loopBody (forward c) 16).drop 3).take 1) s := by
  simp [FPStages.forwardKernel,forward,FPStages.forwardSource,FPStages.fusedForwardSource,
    FPQuery.source,query,FPQuery.fusedSource,
    loopBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    boundaryZero,Option.bind,ComputeDType.eraseDType,withDot]
  cases ha : s.regs .real [c.BTL,c.BV] "b_o" <;>
    cases hr0 : s.regs .real [] "b_b" <;>
    cases hr1 : s.regs .real [c.BTL,c.BTS] "b_s" <;>
    cases hr2 : s.regs .real [c.BTS,c.BV] "b_v" <;>
    simp [ha,hr0,hr1,hr2,Option.bind,State.setReg,overwrite]
  all_goals funext d sh n; split <;> simp_all

theorem forward_local_safe (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig) (cap : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) cap (((loopBody (FPStages.forwardKernel c) 16).drop 3).take 2) s ↔
      Safety.traceSafeList (withDot M D) cap (((loopBody (forward c) 16).drop 3).take 1) s := by
  simp [FPStages.forwardKernel,forward,FPStages.forwardSource,FPStages.fusedForwardSource,
    FPQuery.source,query,FPQuery.fusedSource,
    loopBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    boundaryZero,Option.bind,ComputeDType.eraseDType,withDot,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  cases ha : s.regs .real [c.BTL,c.BV] "b_o" <;>
    cases hr0 : s.regs .real [] "b_b" <;>
    cases hr1 : s.regs .real [c.BTL,c.BTS] "b_s" <;>
    cases hr2 : s.regs .real [c.BTS,c.BV] "b_v" <;>
    simp [ha,hr0,hr1,hr2,Option.bind,State.setReg,overwrite]

theorem query_local_run (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) (s : State α) :
    run (withDot M D) (((loopBody (FPQuery.source c dq) 8).drop 3).take 2) s=
      run (withDot M D) (((loopBody (query c dq) 8).drop 3).take 1) s := by
  simp [FPStages.forwardKernel,forward,FPStages.forwardSource,FPStages.fusedForwardSource,
    FPQuery.source,query,FPQuery.fusedSource,
    loopBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    boundaryZero,Option.bind,ComputeDType.eraseDType,withDot]
  cases ha : s.regs .real [c.BTL,c.BK] "b_dq" <;>
    cases hr0 : s.regs .real [] "d_b" <;>
    cases hr1 : s.regs .real [c.BTL,c.BTS] "b_ds" <;>
    cases hr2 : s.regs .real [c.BTS,c.BK] "b_k" <;>
    simp [ha,hr0,hr1,hr2,Option.bind,State.setReg,overwrite]
  all_goals funext d sh n; split <;> simp_all

theorem query_local_safe (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) (cap : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) cap (((loopBody (FPQuery.source c dq) 8).drop 3).take 2) s ↔
      Safety.traceSafeList (withDot M D) cap (((loopBody (query c dq) 8).drop 3).take 1) s := by
  simp [FPStages.forwardKernel,forward,FPStages.forwardSource,FPStages.fusedForwardSource,
    FPQuery.source,query,FPQuery.fusedSource,
    loopBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    boundaryZero,Option.bind,ComputeDType.eraseDType,withDot,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  cases ha : s.regs .real [c.BTL,c.BK] "b_dq" <;>
    cases hr0 : s.regs .real [] "d_b" <;>
    cases hr1 : s.regs .real [c.BTL,c.BTS] "b_ds" <;>
    cases hr2 : s.regs .real [c.BTS,c.BK] "b_k" <;>
    simp [ha,hr0,hr1,hr2,Option.bind,State.setReg,overwrite]

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
