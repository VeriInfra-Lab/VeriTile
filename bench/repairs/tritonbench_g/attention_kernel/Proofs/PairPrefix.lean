import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOResults
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttentionKernel.Pair
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def kernelAs (storage : TypedSource.Storage) (p : Parameters) : ComputeKernel :=
  TypedSource.queryEarly storage p.Q p.K p.V p.Bias p.Out p.scale
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.bias.batch p.bias.head p.bias.row
    p.batches p.heads p.length 0 64 128 p.dim p.bm p.bn

def kernel (p : Parameters) := kernelAs p.storage p

theorem suffix_eq (storage : TypedSource.Storage) (p : Parameters) :
    (kernelAs storage p).surfaceBody.drop 18 =
      (Source.kernelAs storage p).surfaceBody.drop 18 := by
  cases storage <;> rfl

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α)
    (storage : TypedSource.Storage) (p : Parameters) (s : State α) :
    run M ((kernelAs storage p).surfaceBody.take 18) s =
      run M ((Source.kernelAs storage p).surfaceBody.take 18) s := by
  cases storage
  all_goals
    simp [kernelAs,Source.kernelAs,TypedSource.queryEarly,TypedSource.repaired,
      TypedSource.FP16.queryEarly,TypedSource.FP16.repaired,
      TypedSource.BF16.queryEarly,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,numeric,integralFloorDiv,integralMod,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast]
  all_goals
    congr 1
    simp only [State.setReg]
    congr 1
    funext d sh name
    by_cases hq : name="q"
    · subst name; simp only [String.reduceEq,if_false,if_true]
    by_cases hs : name="qk_scale"
    · subst name; simp only [String.reduceEq,if_false,if_true]
    simp only [hq,hs,if_false]

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (storage : TypedSource.Storage) (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds ((kernelAs storage p).surfaceBody.take 18) s ↔
      Safety.traceSafeList M bounds ((Source.kernelAs storage p).surfaceBody.take 18) s := by
  cases storage
  all_goals
    simp only [kernelAs,Source.kernelAs,TypedSource.queryEarly,TypedSource.repaired,
      TypedSource.FP16.queryEarly,TypedSource.FP16.repaired,
      TypedSource.BF16.queryEarly,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
    simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    simp only [IOSafety.safeBlockPtr2]
    simp [Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]

end
end VeriTile.Bench.Repairs.AttentionKernel.Pair
