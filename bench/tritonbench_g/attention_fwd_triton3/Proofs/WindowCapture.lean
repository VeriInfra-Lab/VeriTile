import bench.tritonbench_g.attention_fwd_triton3.Proofs.Prelude
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

def Captured (c : Call) (o : Ops α) (a : Inputs α) (s : State α) : Prop :=
  s.regs .nat [] "start_m"=some (fun _=>a.tile) ∧
  s.regs .real [] "__veritile_statement_jit_7"=some (fun _=>scale c o) ∧
  s.regs .real [c.BM,c.D] "__veritile_statement_jit_3"=some (query c o a) ∧
  s.regs .ptr [c.BM] "m_ptrs"=some (fun i=>(("M" : RegionName),carryAddress c a i.1.val)) ∧
  s.regs .ptr [c.BM] "l_ptrs"=some (fun i=>(("L" : RegionName),carryAddress c a i.1.val)) ∧
  s.regs .blockPtr [c.BM,c.D] "O_block_ptr"=some (fun _=>optr c a) ∧
  s.regs .real [c.BM] "__veritile_statement_jit_2"=some (initial c o a).center ∧
  s.regs .real [c.BM] "__veritile_statement_jit_1"=some (initial c o a).mass ∧
  s.regs .real [c.BM,c.D] "__veritile_statement_jit_0"=some (initial c o a).weighted ∧
  s.regs .blockPtr [c.D,c.BN] "__veritile_statement_jit_4"=some (fun _=>kptr c a) ∧
  s.regs .blockPtr [c.BN,c.D] "__veritile_statement_jit_5"=some (fun _=>vptr c a) ∧
  s.regs .int [] "__veritile_statement_jit_6"=some (fun _=>(a.tile : Int)) ∧
  s.regs .nat [] "__veritile_statement_jit_8"=some (fun _=>c.NK) ∧
  s.regs .int [] "__veritile_statement_jit_9"=some (fun _=>c.offset) ∧
  s.regs .int [] "__veritile_statement_jit_10"=some (fun _=>c.size)

theorem capture (c : Call) (o : Ops α) (initialState s : State α)
    (live : Started c o (inputs initialState) s) (memory : s.mem=initialState.mem) :
    ∃ t,run (engine o.algebra) (((parts c).before.drop 21).take 12) s=some t ∧
      Captured c o (inputs initialState) t ∧ t.mem=initialState.mem := by
  rcases live with ⟨⟨h0,h1,h2,h3,h4,h5,h6,hscale⟩,h7,h8,h9,h10,h11⟩
  simp (config := {maxSteps := 250000}) [parts,Kernels.source,Quoted.kernel,
    DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,sourceKernel,
    ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
    ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
    FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
    natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,
    h0,h1,h2,h3,h4,h5,h6,hscale,h7,h8,h9,h10,h11,Captured,inputs,memory]

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
