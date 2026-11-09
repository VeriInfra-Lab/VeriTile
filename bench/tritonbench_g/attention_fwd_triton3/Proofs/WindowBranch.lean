import bench.tritonbench_g.attention_fwd_triton3.Proofs.WindowCapture
import VeriTile.Triton.Float.Integer
import VeriTile.Triton.Float.SignedCast

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem lower_clamp (x : Int) : max x 0=if x<0 then 0 else x := by split_ifs <;> omega
theorem upper_clamp (x cap : Int) : min x cap=if cap<x then cap else x := by split_ifs <;> omega
attribute [local simp 1100] FP.Structural.eval_truncDiv FP.Structural.eval_intToSigned

theorem window_branch (c : Call) (o : Ops α) (initialState s : State α)
    (live : Captured c o (inputs initialState) s) (memory : s.mem=initialState.mem) :
    ∃ t,run (engine o.algebra) ((parts c).before.drop 33) s=some t ∧
      Matches c o (inputs initialState) (initial c o (inputs initialState)) t ∧
      t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h13,h14,h15,h16⟩
  by_cases hw : c.window=0 <;> by_cases hc : c.complement=0
  all_goals
    by_cases hl : (((initialState.pids 0 : Int)*c.BM+c.offset-c.size+1).tdiv c.BN*c.BN)<0
    all_goals
      by_cases hh : (c.NK : Int)<((((initialState.pids 0 : Int)+1)*c.BM-1+c.offset+c.BN).tdiv c.BN*c.BN)
      all_goals
        simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
          DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
          evalDeltas,Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
          FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
          natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,boundaryZero,
          h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h13,h14,h15,h16,hw,hc,hl,hh,
          Matches,initial,lower,upper,lower_clamp,upper_clamp,inputs,
          Int.natCast_mul,Int.natCast_add,bop,Broadcast.leftIndex,Broadcast.rightIndex,memory]
  all_goals simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex]

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
