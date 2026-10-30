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

theorem entry_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts p).before s := by
  have hq:=hb.query
  have hbi:=hb.bias
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals
    simp only [parts,partsAs,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,↓reduceIte]
    simp only [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
    simp only [safeBlockPtr2]
    simp [safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]
  all_goals
    constructor
    · intro i j hi
      change (decide (s.pids 0*bm+i.val<length) && Bool.true)=Bool.true at hi
      simp only [Bool.and_true,decide_eq_true_eq] at hi
      simpa [base,Nat.add_assoc] using hq _ hi j.val j.isLt
    · intro i j hi
      simpa [base,Nat.add_assoc] using hbi _ hi (j.val%64+64)
        (by have:=Nat.mod_lt j.val (by decide : 0<64);omega)

end
end VeriTile.Bench.Repairs.AttentionKernel.IOSafety
