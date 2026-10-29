import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IOSafetyEntry

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Projection of the actual load-free loop suffix, with no source reconstruction. -/
private def bodyStmt (p : Parameters) (i : Fin 14) : ComputeStmt :=
  ((parts p).body[2+i.val]?).getD (.effectMarker "tl.debug_barrier")

private theorem bodySource (p : Parameters) :
    (parts p).body.drop 2 = List.ofFn (bodyStmt p) := by rfl

theorem body_tail_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M dot)) bounds ((parts p).body.drop 2) s := by
  apply traceSafe_of_pointwise
  intro st member current
  rw [bodySource] at member
  obtain ⟨i,rfl⟩ := List.mem_ofFn.mp member
  fin_cases i
  all_goals first
    | change Safety.stepSafe (engine (withDot M dot)) bounds (.assign _ _ _ _) current
    | change Safety.stepSafe (engine (withDot M dot)) bounds (.ifThen (.constBool p.IS_CAUSAL) [_]) current
  all_goals
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      traceSafe_singleton,evalOp_unfold,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem body_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial scale row a current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).body
      (current.setReg "start_n" .nat [] (fun _ => row)) := by
  have source : (parts p).body =
      .assign .real [p.DIM,64] "k"
        (.compute (.load .fp32 (.blockPtr (.ref .blockPtr [p.DIM,64] "K_block_ptr") [1]) .none .zero)) ::
      .assign .real [64,p.DIM] "v"
        (.compute (.load .fp32 (.blockPtr (.ref .blockPtr [64,p.DIM] "V_block_ptr") [0]) .none .zero)) ::
      (parts p).body.drop 2 := by
    rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
    cases causal <;> rfl
  rw [source,Safety.traceSafeList_cons]
  constructor
  · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,
      State.setReg,ctx.keyPointer,true_and]
    intro ptr hp
    have pe := Option.some.inj hp
    subst ptr
    intro i _ hi
    have active : row+i.2.1.val < p.SEQLEN := by
      change (decide (row+i.2.1.val < p.SEQLEN) && Bool.true)=Bool.true at hi
      simpa using hi
    have addr : (kptr p initial row).address (TileShape.indexToList [p.DIM,64] i) =
        base p initial p.stride_k_bs p.stride_k_head+i.1.val*p.stride_k_dim+
          (row+i.2.1.val)*p.stride_k_seqlen := by
      change base p initial p.stride_k_bs p.stride_k_head +
        ((0+i.1.val)*p.stride_k_dim+(row+i.2.1.val)*p.stride_k_seqlen+0)=_
      simp [Nat.add_assoc]
    rw [addr]
    exact hb.key row i active
  · intro afterK executedK
    have vp : afterK.regs .blockPtr [64,p.DIM] "V_block_ptr" =
        some (fun _ => vptr p initial row) := by
      rw [assign_keeps_reg _ _ _ _ _ _ _ "V_block_ptr" (by decide) _ _ executedK]
      simpa [State.setReg] using ctx.valuePointer
    rw [Safety.traceSafeList_cons]
    constructor
    · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,
        vp,true_and]
      intro ptr hp
      have pe := Option.some.inj hp
      subst ptr
      intro i _ hi
      have active : row+i.1.val < p.SEQLEN := by
        change (decide (row+i.1.val < p.SEQLEN) && Bool.true)=Bool.true at hi
        simpa using hi
      have addr : (vptr p initial row).address (TileShape.indexToList [64,p.DIM] i) =
          base p initial p.stride_v_bs p.stride_v_head+(row+i.1.val)*p.stride_v_seqlen+
            i.2.1.val*p.stride_v_dim := by
        change base p initial p.stride_v_bs p.stride_v_head +
          ((row+i.1.val)*p.stride_v_seqlen+(0+i.2.1.val)*p.stride_v_dim+0)=_
        simp [Nat.add_assoc]
      rw [addr]
      exact hb.value row i active
    · intro afterV _
      exact body_tail_safe M dot p afterV bounds

#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
