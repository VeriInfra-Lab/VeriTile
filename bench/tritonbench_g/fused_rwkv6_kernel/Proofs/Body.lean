import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.BodyExec
import bench.tritonbench_g.attention_forward_triton.Proofs.Scatter

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
open AttentionForwardTriton.Scatter (masked_scatter_memory masked_scatter_regs)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem body_context (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (n : Nat) (ctx : Context M c initial scale n current) (live : n<c.T) :
    Context M c initial scale (n+1) (bodyExec M c initial current scale n ctx).val := by
  by_cases hr : c.reverse=Bool.true
  all_goals constructor
  all_goals simp [bodyExec,hr,ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,
    ctx.pq,ctx.pk,ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,
    masked_scatter_memory,masked_scatter_regs,
    loopMemory,rowWrite,rowValues,terms,carry,scalarState,scalarTerm,scalarOuter,
    ops,memory,regionName,inputs,inputKey,inputValue,biasAddress,
    bias,keyMask,valueMask,pointer_next,pointer_live _ _ _ _ live,
    keyBase,valueBase,outputBase,loopMemory_input,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,
    show (0.0 : ℝ)=0 by norm_num]
  all_goals try funext i
  all_goals try simp [pointer,hr,Source.carry,Source.terms,scalarState,scalarTerm,scalarOuter,
    ops,memory,regionName,inputs,inputKey,inputValue,biasAddress,Source.bias,keyMask,valueMask,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,Nat.sub_sub]
  all_goals try rfl
  all_goals
    apply congrFun
    apply congrArg (fun step => List.foldl step (loopMemory M c initial scale n)
      (TileShape.allIndices [c.BV]))
    funext mem lane
    split_ifs
    · funext r b
      split_ifs
      · apply congrArg (fun v => Cell.mk .real (M.cast (some .fp32) .real .real v))
        apply congrArg (fun xs : Values α .real [c.BV,c.BK] =>
          M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false xs lane)
        funext j
        simp [Source.terms,Source.carry,FP.Structural.bop,scalarTerm,scalarOuter,
          ops,memory,regionName,inputs,inputKey,inputValue,biasAddress,
          Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
          Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
      · rfl
    · rfl

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
