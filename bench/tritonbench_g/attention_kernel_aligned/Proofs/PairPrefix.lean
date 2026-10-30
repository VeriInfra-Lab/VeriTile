import bench.tritonbench_g.attention_kernel_aligned.Proofs.IOSafety
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16IOSafety
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Pair
open VeriTile Triton FP.Structural Full
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def originalKernel (bf16 : Bool) (p : Parameters) : ComputeKernel :=
  if bf16 then BF16.Source.source p else Source.source p

def candidateKernel (bf16 : Bool) (p : Parameters) : ComputeKernel :=
  if bf16 then BF16.Source.queryEarly p else Source.queryEarly p

theorem suffix_eq (bf16 : Bool) (p : Parameters) :
    (candidateKernel bf16 p).surfaceBody.drop 18 =
      (originalKernel bf16 p).surfaceBody.drop 18 := by
  cases bf16 <;> rfl

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α)
    (bf16 : Bool) (p : Parameters) (s : State α) :
    run M ((candidateKernel bf16 p).surfaceBody.take 18) s =
      run M ((originalKernel bf16 p).surfaceBody.take 18) s := by
  cases bf16
  all_goals
    simp [candidateKernel,originalKernel,Source.queryEarly,Source.source,
      BF16.Source.queryEarly,BF16.Source.source,
      Source.typedSurfaceQueryEarly,Source.typedSurface,
      BF16.Source.typedSurfaceQueryEarly,BF16.Source.typedSurface,
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
    (bf16 : Bool) (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds ((candidateKernel bf16 p).surfaceBody.take 18) s ↔
      Safety.traceSafeList M bounds ((originalKernel bf16 p).surfaceBody.take 18) s := by
  cases bf16
  all_goals
    simp only [candidateKernel,originalKernel,Bool.false_eq_true,if_false,if_true,
      Source.queryEarly,Source.source,BF16.Source.queryEarly,BF16.Source.source,
      Source.typedSurfaceQueryEarly,Source.typedSurface,
      BF16.Source.typedSurfaceQueryEarly,BF16.Source.typedSurface,
      ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
    simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    simp only [SourceSafety.safeBlockPtr2]
    simp [Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]

#axiomsClean prefix_run
#axiomsClean prefix_safe
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Pair
