import bench.tritonbench_g.flash_attn.Proofs.IOSafetyPrelude

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem assign_value (M : Algebra α) (dtype shape name) (expr : ComputeExpr dtype shape)
    (s t : State α) (v : Values α dtype shape)
    (he : evalExpr M expr s=some v)
    (h : step M (.assign dtype shape name expr) s=some t) :
    t.regs dtype shape name=some v := by
  simp only [step,he,bind,pure,Option.bind_some,Option.some.injEq] at h
  subst t
  simp

theorem assign_keeps_pids (M : Algebra α) (dtype shape name) (expr : ComputeExpr dtype shape)
    (s t : State α) (h : step M (.assign dtype shape name expr) s=some t) :
    t.pids=s.pids := by
  simp only [step] at h
  cases he : evalExpr M expr s with
  | none => simp [he] at h
  | some value =>
    simp only [he,bind,pure,Option.bind_some,Option.some.injEq] at h
    subst t
    rfl

/-- Only the pointer-relevant register values are propagated. The existing
source execution supplies the intermediate states; no attention computation is
reconstructed in this safety proof. -/
theorem preQuery_pointer (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (s t : State α)
    (executed : run (engine (withDot M dot)) (preQuery p) (bindScale scale s)=some t) :
    t.regs .blockPtr [128,p.DIM] "Q_block_ptr"=some (fun _ => qptr p s) := by
  change run _ [
    .assign .nat [] "start_m" (.alg (.programId 0)),
    .assign .nat [] "off_bs_head" (.alg (.programId 1)),
    .assign .nat [] "off_bs" (.alg (.floorDiv .nat .nil (.ref .nat [] "off_bs_head") (.constNat p.HEAD))),
    .assign .nat [] "off_head" (.alg (.mod .nat .nil (.ref .nat [] "off_bs_head") (.constNat p.HEAD))),
    .assign .nat [] "q_base_offset" (.alg (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "off_bs") (.constNat p.stride_q_bs))
      (.mul .nat .nil (.ref .nat [] "off_head") (.constNat p.stride_q_head)))),
    .assign .nat [] "k_base_offset" (.alg (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "off_bs") (.constNat p.stride_k_bs))
      (.mul .nat .nil (.ref .nat [] "off_head") (.constNat p.stride_k_head)))),
    .assign .nat [] "v_base_offset" (.alg (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "off_bs") (.constNat p.stride_v_bs))
      (.mul .nat .nil (.ref .nat [] "off_head") (.constNat p.stride_v_head)))),
    .assign .nat [] "o_base_offset" (.alg (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "off_bs") (.constNat p.stride_o_bs))
      (.mul .nat .nil (.ref .nat [] "off_head") (.constNat p.stride_o_head)))),
    .assign .blockPtr [128,p.DIM] "Q_block_ptr" (.alg
      (.makeBlockPtrDynOffsets "Q" (.ref .nat [] "q_base_offset")
        [p.SEQLEN,p.DIM] [128,p.DIM] [p.stride_q_seqlen,p.stride_q_dim]
        [.mul .nat .nil (.ref .nat [] "start_m") (.constNat 128),.constNat 0]))]
    (bindScale scale s)=some t at executed
  simp only [run,bind,pure,Option.bind_eq_some_iff,Option.some.injEq] at executed
  rcases executed with ⟨s1,h1,s2,h2,s3,h3,s4,h4,s5,h5,s6,h6,s7,h7,s8,h8,s9,h9,he⟩
  subst s9
  have start1 : s1.regs .nat [] "start_m"=some (fun _ => s.pids 0) := by
    apply assign_value _ _ _ _ _ _ _ _ _ h1
    rfl
  have pid1 : s1.pids=s.pids := by
    change s1.pids=(bindScale scale s).pids
    exact assign_keeps_pids _ _ _ _ _ _ _ h1
  have head2 : s2.regs .nat [] "off_bs_head"=some (fun _ => s.pids 1) := by
    apply assign_value _ _ _ _ _ _ _ _ _ h2
    simp only [evalExpr,evalOp_unfold,pid1]
  have batch3 : s3.regs .nat [] "off_bs"=some (fun _ => s.pids 1/p.HEAD) := by
    apply assign_value _ _ _ _ _ _ _ _ _ h3
    simp [evalExpr,evalOp_unfold,head2,integralFloorDiv,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl
  have head3 : s3.regs .nat [] "off_bs_head"=some (fun _ => s.pids 1) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "off_bs_head" (by decide) _ _ h3]
    exact head2
  have mod4 : s4.regs .nat [] "off_head"=some (fun _ => s.pids 1%p.HEAD) := by
    apply assign_value _ _ _ _ _ _ _ _ _ h4
    simp [evalExpr,evalOp_unfold,head3,integralMod,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl
  have batch4 : s4.regs .nat [] "off_bs"=some (fun _ => s.pids 1/p.HEAD) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "off_bs" (by decide) _ _ h4]
    exact batch3
  have base5 : s5.regs .nat [] "q_base_offset"=some (fun _ => base p s p.stride_q_bs p.stride_q_head) := by
    apply assign_value _ _ _ _ _ _ _ _ _ h5
    simp [evalExpr,evalOp_unfold,batch4,mod4,numeric,FP.Structural.bop,base,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl
  have base8 : s8.regs .nat [] "q_base_offset"=some (fun _ => base p s p.stride_q_bs p.stride_q_head) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "q_base_offset" (by decide) _ _ h8,
      assign_keeps_reg _ _ _ _ _ _ _ "q_base_offset" (by decide) _ _ h7,
      assign_keeps_reg _ _ _ _ _ _ _ "q_base_offset" (by decide) _ _ h6]
    exact base5
  have start8 : s8.regs .nat [] "start_m"=some (fun _ => s.pids 0) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h8,
      assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h7,
      assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h6,
      assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h5,
      assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h4,
      assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h3,
      assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h2]
    exact start1
  apply assign_value _ _ _ _ _ _ _ _ _ h9
  simp [evalExpr,evalOp_unfold,base8,start8,evalOffsets,numeric,FP.Structural.bop,qptr,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

#axiomsClean preQuery_pointer
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
