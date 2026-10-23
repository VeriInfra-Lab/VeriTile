import bench.repairs.tritonbench_g.parallel_attention.Proofs.FusedContext

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def forward (c : ForwardConfig) : ComputeKernel :=
  fusedForwardSource c.q c.k c.v c.o c.z c.s_qk_h c.s_qk_t c.s_qk_d
    c.s_vo_h c.s_vo_t c.s_vo_d c.scale c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def query (c : QueryConfig) : ComputeKernel :=
  fusedQuerySource c.q c.k c.v c.do_ c.dz c.dq c.i_bh c.i_c c.i_k c.i_v
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def shell (c : QueryConfig) : ComputeKernel :=
  shellSource c.q c.k c.v c.do_ c.dz c.dq c.dk c.dv c.i_bh c.i_c c.i_k c.i_v c.i_h
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def fusedShell (c : QueryConfig) : ComputeKernel :=
  fusedShellSource c.q c.k c.v c.do_ c.dz c.dq c.dk c.dv c.i_bh c.i_c c.i_k c.i_v c.i_h
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def loopBody (k : ComputeKernel) (n : Nat) : List ComputeStmt :=
  let code : List ComputeStmt := k.surfaceBody
  match code[n]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ b)=>b
  | _=>[]

theorem overwrite (s : State α) (n : RegName) (d : TileDType) (sh : TileShape)
    (u v : Values α d sh) : (s.setReg n d sh u).setReg n d sh v=s.setReg n d sh v := by
  simp only [State.setReg]
  congr 1
  funext d' sh' n'
  split <;> simp_all

theorem forward_local_run (M : Algebra α) (c : ForwardConfig) (s : State α) :
    run M (((loopBody (forwardKernel c) 19).drop 4).take 2) s =
      run M (((loopBody (forward c) 19).drop 4).take 1) s := by
  simp [loopBody,forwardKernel,forward,forwardSource,fusedForwardSource,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType]
  cases hx : s.regs .real [c.BTL,c.BTS] "b_s" <;>
    cases hm : s.regs .bool [c.BTL,c.BTS] "m_s" <;>
    simp [hx,hm,Option.bind,State.setReg,overwrite]
  funext d sh n; split <;> simp_all

theorem forward_local_safe (M : Algebra α) (c : ForwardConfig) (cap : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M cap (((loopBody (forwardKernel c) 19).drop 4).take 2) s ↔
      Safety.traceSafeList M cap (((loopBody (forward c) 19).drop 4).take 1) s := by
  simp [loopBody,forwardKernel,forward,forwardSource,fusedForwardSource,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType]
  cases hx : s.regs .real [c.BTL,c.BTS] "b_s" <;>
    cases hm : s.regs .bool [c.BTL,c.BTS] "m_s" <;>
    simp [hx,hm,Option.bind,State.setReg]

theorem query_local_run (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (s : State α) :
    run (withDot M D) (((loopBody (queryKernel c) 16).drop 6).take 2) s =
      run (withDot M D) (((loopBody (query c) 16).drop 6).take 1) s := by
  simp [loopBody,queryKernel,query,querySource,fusedQuerySource,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,withDot]
  cases hq : s.regs .real [c.BTL,c.BK] "b_q" <;>
    cases hk : s.regs .real [c.BTS,c.BK] "b_k" <;>
    cases hm : s.regs .bool [c.BTL,c.BTS] "m_s" <;>
    simp [hq,hk,hm,Option.bind,State.setReg]
  funext d sh n; split <;> simp_all

theorem query_local_safe (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (cap : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) cap (((loopBody (queryKernel c) 16).drop 6).take 2) s ↔
      Safety.traceSafeList (withDot M D) cap (((loopBody (query c) 16).drop 6).take 1) s := by
  simp [loopBody,queryKernel,query,querySource,fusedQuerySource,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,withDot]
  cases hq : s.regs .real [c.BTL,c.BK] "b_q" <;>
    cases hk : s.regs .real [c.BTS,c.BK] "b_k" <;>
    cases hm : s.regs .bool [c.BTL,c.BTS] "m_s" <;>
    simp [hq,hk,hm,Option.bind,State.setReg]


end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
