import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOSafetyBasics

namespace VeriTile.Bench.Repairs.AttentionKernel.IOSafety
open VeriTile Triton FP.Structural
open VeriTile.Triton.FP.ProfiledRewrite (engine)
open Whole001 Source Whole Machine
open FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem post_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial row a current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts p).after current := by
  have hout:=hb.output
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals
    simp only [parts,partsAs,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,↓reduceIte]
    simp only [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
    simp only [safeBlockPtr2]
    simp [safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      ctx.startM,ctx.outputOffset,ctx.normalizer,ctx.value,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]
    constructor
    · intro i j hi
      change (decide (initial.pids 0*bm+i.val<length) && Bool.true)=Bool.true at hi
      simp only [Bool.and_true,decide_eq_true_eq] at hi
      simpa [base,Nat.add_assoc] using hout _ hi j.val j.isLt
    · split <;> trivial

end
end VeriTile.Bench.Repairs.AttentionKernel.IOSafety
