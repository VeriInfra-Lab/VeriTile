import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOImplements
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem suffix_eq (c : Shape) :
    (earlyUKernel c).surfaceBody.drop 16 = (kernel c).surfaceBody.drop 16 := by rfl

/-- All original registers and the complete memory agree before entering the
unchanged recurrent loop, for arbitrary payloads and scalar primitives. -/
theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (s : State α) :
    run M ((earlyUKernel c).surfaceBody.take 16) s =
      run M ((kernel c).surfaceBody.take 16) s := by
  by_cases hi : c.initial = Bool.true
  all_goals
    simp [earlyUKernel,kernel,ComputeKernel.surfaceBody,hi,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,ComputeDType.eraseDType,
      numeric,integralMod,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast]
    simp only [State.setReg]
    congr 1
    funext d sh name
    by_cases hp : name="p_u"
    · subst name; simp only [String.reduceEq,if_false,if_true]
    by_cases hm : name="mask_bk"
    · subst name; simp only [String.reduceEq,if_false,if_true]
    by_cases hu : name="b_u"
    · subst name; simp only [String.reduceEq,if_false,if_true]
    simp only [hp,hm,hu,if_false]

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds ((earlyUKernel c).surfaceBody.take 16) s ↔
      Safety.traceSafeList M bounds ((kernel c).surfaceBody.take 16) s := by
  by_cases hi : c.initial = Bool.true
  all_goals
    simp [earlyUKernel,kernel,ComputeKernel.surfaceBody,hi,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,ComputeDType.eraseDType,
      numeric,integralMod,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.indexToList,and_comm]

#axiomsClean prefix_run
#axiomsClean prefix_safe
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
