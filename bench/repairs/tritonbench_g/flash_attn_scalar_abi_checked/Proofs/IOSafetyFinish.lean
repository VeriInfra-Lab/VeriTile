import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IOSafetyBody

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Index the five existing post-loop statements, without a duplicate AST. -/
def post (p : Parameters) (i : Fin 5) : ComputeStmt :=
  (parts p).after[i.val]'(by change i.val<5;exact i.isLt)

theorem post_list (p : Parameters) :
    (parts p).after=[post p 0,post p 1,post p 2,post p 3,post p 4] := rfl

theorem post_pure (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat)
    (s : State α) (i : Fin 5) (hi : i=0 ∨ i=1 ∨ i=3) :
    Safety.stepSafe M bounds (post p i) s := by
  rcases hi with rfl|rfl|rfl
  all_goals
    change Safety.stepSafe M bounds (.assign _ _ _ _) s
  all_goals
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]

theorem ptr_store_keeps_regs (M : Algebra α) (dtype : TileDType) (shape : TileShape)
    (pointer : Op .ptr shape) (condition : Op .bool shape) (value : ComputeExpr dtype shape)
    (s t : State α) (addresses : TileIndex shape → RegionName × Nat)
    (active : TileIndex shape → Bool)
    (hp : evalOp M none pointer s=some addresses)
    (ha : evalOp M none condition s=some active)
    (he : step M (.store dtype shape (.ptr pointer) value (.mask condition)) s=some t) :
    t.regs=s.regs := by
  simp only [step] at he
  cases hv : evalExpr M value s with
  | none => simp [hv] at he
  | some values =>
    simp only [hv,bind,pure,Option.bind_some,store,hp,ha,Option.some.injEq] at he
    subst t
    have foldregs : ∀ (items : List (TileIndex shape)) (initial : State α),
        (items.foldl (fun current i => if active i then
          current.write (addresses i).1 (addresses i).2 (.mk dtype (values i)) else current) initial).regs=initial.regs := by
      intro items
      induction items with
      | nil => intro initial;rfl
      | cons i rest ih =>
        intro initial
        simp only [List.foldl_cons,ih]
        split <;> rfl
    exact foldregs _ _

theorem finish_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial scale row a current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).after current := by
  rw [post_list,Safety.traceSafeList_cons]
  refine ⟨post_pure _ p bounds current 0 (Or.inl rfl),?_⟩
  intro s1 h1
  rw [Safety.traceSafeList_cons]
  refine ⟨post_pure _ p bounds s1 1 (Or.inr (Or.inl rfl)),?_⟩
  intro s2 h2
  have head1 : s1.regs .nat [] "off_bs_head"=some (fun _ => initial.pids 1) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "off_bs_head" (by decide) _ _ h1]
    exact ctx.head
  have rows1 : s1.regs .nat [128] "off_m"=some (fun i => initial.pids 0*128+i.1.val) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "off_m" (by decide) _ _ h1]
    exact ctx.rows
  have rows2 : s2.regs .nat [128] "off_m"=some (fun i => initial.pids 0*128+i.1.val) := by
    rw [assign_keeps_reg _ _ _ _ _ _ _ "off_m" (by decide) _ _ h2]
    exact rows1
  have pointers : s2.regs .ptr [128] "l_ptr"=some (fun i => (("L" : RegionName),lAddress p initial i)) := by
    apply assign_value _ _ _ _ _ _ _ _ _ h2
    simp [evalExpr,evalOp_unfold,head1,rows1,numeric,FP.Structural.bop,lAddress,
      Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    funext i
    change (("L" : RegionName),0+(initial.pids 1*p.SEQLEN+(initial.pids 0*128+i.1.val)))=_
    simp
  have mask_eval : evalOp (engine (withDot M dot)) none
      (.lt .nat .scalarR (.ref .nat [128] "off_m") (.constNat p.SEQLEN)) s2=
        some (fun i => decide (lActive p initial i)) := by
    simp [evalOp_unfold,rows2,numericLt,natLt,FP.Structural.bop,lActive,
      Broadcast.leftIndex,Broadcast.rightIndex]
    rfl
  rw [Safety.traceSafeList_cons]
  constructor
  · change Safety.stepSafe (engine (withDot M dot)) bounds
      (.store .real [128] (.ptr (.ref .ptr [128] "l_ptr")) _
        (.mask (.lt .nat .scalarR (.ref .nat [128] "off_m") (.constNat p.SEQLEN)))) s2
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.safeOp_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,pointers,true_and]
    intro ptr he
    have pe := Option.some.inj he
    subst ptr
    intro i active
    rcases active with ⟨mask,hm,hi⟩
    have me := Option.some.inj (mask_eval.symm.trans hm)
    subst mask
    exact hb.log i (by simpa using hi)
  · intro s3 h3
    have kept : s3.regs=s2.regs := by
      exact ptr_store_keeps_regs (engine (withDot M dot)) .real [128]
        (.ref .ptr [128] "l_ptr")
        (.lt .nat .scalarR (.ref .nat [128] "off_m") (.constNat p.SEQLEN)) _ s2 s3
        (fun i => (("L" : RegionName),lAddress p initial i))
        (fun i => decide (lActive p initial i)) pointers mask_eval h3
    rw [Safety.traceSafeList_cons]
    refine ⟨post_pure _ p bounds s3 3 (Or.inr (Or.inr rfl)),?_⟩
    intro s4 h4
    have start3 : s3.regs .nat [] "start_m"=some (fun _ => initial.pids 0) := by
      rw [kept,
        assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h2,
        assign_keeps_reg _ _ _ _ _ _ _ "start_m" (by decide) _ _ h1]
      exact ctx.startM
    have offset3 : s3.regs .nat [] "o_base_offset"=some (fun _ => base p initial p.stride_o_bs p.stride_o_head) := by
      rw [kept,
        assign_keeps_reg _ _ _ _ _ _ _ "o_base_offset" (by decide) _ _ h2,
        assign_keeps_reg _ _ _ _ _ _ _ "o_base_offset" (by decide) _ _ h1]
      exact ctx.outputOffset
    have outputPointer : s4.regs .blockPtr [128,p.DIM] "O_block_ptr"=some (fun _ => optr p initial) := by
      apply assign_value _ _ _ _ _ _ _ _ _ h4
      simp [evalExpr,evalOp_unfold,start3,offset3,evalOffsets,numeric,FP.Structural.bop,optr,
        Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    rw [traceSafe_singleton]
    change Safety.stepSafe (engine (withDot M dot)) bounds
      (.store .real [128,p.DIM] (.blockPtr (.ref .blockPtr [128,p.DIM] "O_block_ptr") [0]) _ .none) s4
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.safeOp_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,outputPointer,true_and]
    intro ptr he
    have pe := Option.some.inj he
    subst ptr
    intro i _ hi
    have active : active p initial i := by
      simpa [optr,check0,active] using hi
    have addr : (optr p initial).address (TileShape.indexToList [128,p.DIM] i)=address p initial i := by
      change base p initial p.stride_o_bs p.stride_o_head+
        ((initial.pids 0*128+i.1.val)*p.stride_o_seqlen+(0+i.2.1.val)*p.stride_o_dim+0)=_
      simp [address,Nat.add_assoc]
    rw [addr]
    exact hb.output i active

#axiomsClean finish_safe
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
