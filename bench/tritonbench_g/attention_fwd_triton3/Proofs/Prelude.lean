import bench.tritonbench_g.attention_fwd_triton3.Proofs.Reference
import VeriTile.Triton.Float.Integer

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
attribute [local simp 1100] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod

variable {α : Type} [Inhabited α]

def HeaderMatches (c : Call) (o : Ops α) (a : Inputs α) (s : State α) : Prop :=
  s.regs .nat [] "start_m"=some (fun _ => a.tile) ∧
  s.regs .ptr [c.BM] "m_ptrs"=some (fun i : TileIndex [c.BM] => (("M" : RegionName),carryAddress c a i.1.val)) ∧
  s.regs .ptr [c.BM] "l_ptrs"=some (fun i : TileIndex [c.BM] => (("L" : RegionName),carryAddress c a i.1.val)) ∧
  s.regs .blockPtr [c.BM,c.D] "O_block_ptr"=some (fun _ => optr c a) ∧
  s.regs .blockPtr [c.BM,c.D] "Q_block_ptr"=some (fun _ => qptr c a) ∧
  s.regs .blockPtr [c.D,c.BN] "K_block_ptr"=some (fun _ => kptr c a) ∧
  s.regs .blockPtr [c.BN,c.D] "V_block_ptr"=some (fun _ => vptr c a) ∧
  s.regs .real [] "sm_scale"=some (fun _ => o.scalar.literal (some .fp32) .real c.scale)

theorem header (c : Call) (o : Ops α) (s : State α) :
    ∃ t,run (engine o.algebra) ((parts c).before.take 17) s=some t ∧
      HeaderMatches c o (inputs s) t ∧ t.mem=s.mem := by
  simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
    DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
    FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
    natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,
    HeaderMatches,inputs,qptr,kptr,vptr,optr,carryAddress,row,batch,head,kvhead,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]
  simp only [←Int.ofNat_tdiv,←Int.ofNat_tmod,←Int.natCast_ediv,←Int.natCast_add,
    ←Int.natCast_mul,Int.toNat_natCast]
  simp

def Started (c : Call) (o : Ops α) (a : Inputs α) (s : State α) : Prop :=
  HeaderMatches c o a s ∧
  s.regs .real [] "qk_scale"=some (fun _ => scale c o) ∧
  s.regs .real [c.BM,c.D] "q"=some (query c o a) ∧
  s.regs .real [c.BM] "m_i"=some (initial c o a).center ∧
  s.regs .real [c.BM] "l_i"=some (initial c o a).mass ∧
  s.regs .real [c.BM,c.D] "acc"=some (initial c o a).weighted

theorem initialization (c : Call) (o : Ops α) (initialState s : State α)
    (live : HeaderMatches c o (inputs initialState) s) (memory : s.mem=initialState.mem) :
    ∃ t,run (engine o.algebra) (((parts c).before.drop 17).take 4) s=some t ∧
      Started c o (inputs initialState) t ∧ t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,hscale⟩
  by_cases hi : c.init=0 <;> by_cases he : c.params.evenM=0
  all_goals
    simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
      DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
      FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
      natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,hscale,hi,he,Started,HeaderMatches,initial,scale,query,loadBlock,
      memory,inputs,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp [loadBlock,funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try simp only [←Int.ofNat_tdiv,←Int.ofNat_tmod,←Int.natCast_ediv,←Int.natCast_add,
    ←Int.natCast_mul,Int.toNat_natCast]
  all_goals rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
