import bench.tritonbench_g.flash_attn.Proofs.IOSafetyQueryPointer

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private def assignsOther (name : RegName) : ComputeStmt → Prop
  | .assign _ _ n _ => name ≠ n
  | _ => False

private theorem assignsOther_keeps (M : Algebra α) (st : ComputeStmt) (name : RegName)
    (h : assignsOther name st) (s t : State α) (d : TileDType) (sh : TileShape)
    (he : step M st s=some t) : t.regs d sh name=s.regs d sh name := by
  cases st <;> simp only [assignsOther] at h
  case assign dtype shape n e => exact assign_keeps_reg M dtype shape n e s t name h d sh he

private theorem traceSafe_reg (M : Algebra α) (bounds : RegionName → Nat)
    (d : TileDType) (sh : TileShape) (name : RegName) (value : Values α d sh)
    (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ s : State α, s.regs d sh name=some value → Safety.stepSafe M bounds st s)
    (keeps : ∀ st ∈ code, assignsOther name st)
    (s : State α) (hv : s.regs d sh name=some value) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s hv)
    intro t ht
    apply ih (fun st h => safe st (by simp [h])) (fun st h => keeps st (by simp [h])) t
    rw [assignsOther_keeps M st name (keeps st (by simp)) s t d sh ht]
    exact hv

/-- Index the actual twelve entry-tail statements without reconstructing source. -/
private def entryStmt (p : Parameters) (i : Fin 12) : ComputeStmt :=
  ((parts p).before[9+i.val]?).getD (.effectMarker "tl.debug_barrier")

private theorem entrySource (p : Parameters) :
    (parts p).before.drop 9 = List.ofFn (entryStmt p) := by rfl

theorem entry_tail_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (hp : current.regs .blockPtr [128,p.DIM] "Q_block_ptr"=
      some (fun _ => qptr p initial))
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds ((parts p).before.drop 9) current := by
  have qsafe : ∀ s : State α, s.regs .blockPtr [128,p.DIM] "Q_block_ptr"=some (fun _ => qptr p initial) →
      Safety.stepSafe (engine (withDot M dot)) bounds
        (.assign .real [128,p.DIM] "q"
          (.compute (.load .fp32 (.blockPtr (.ref .blockPtr [128,p.DIM] "Q_block_ptr") [0]) .none .zero))) s := by
    intro s hq
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,hq,true_and]
    intro ptr he
    have pe := Option.some.inj he
    subst ptr
    intro i _ hi
    have active : initial.pids 0*128+i.1.val < p.SEQLEN := by
      simpa [qptr,check0] using hi
    have addr : (qptr p initial).address (TileShape.indexToList [128,p.DIM] i) =
        base p initial p.stride_q_bs p.stride_q_head +
          (initial.pids 0*128+i.1.val)*p.stride_q_seqlen+i.2.1.val*p.stride_q_dim := by
      change base p initial p.stride_q_bs p.stride_q_head +
        ((initial.pids 0*128+i.1.val)*p.stride_q_seqlen+(0+i.2.1.val)*p.stride_q_dim+0) = _
      simp [Nat.add_assoc]
    rw [addr]
    exact hb.query i active
  apply traceSafe_reg _ bounds .blockPtr [128,p.DIM] "Q_block_ptr" (fun _ => qptr p initial) _ _ _ current hp
  · intro st member s hq
    rw [entrySource] at member
    obtain ⟨i,rfl⟩ := List.mem_ofFn.mp member
    fin_cases i
    all_goals first
      | exact qsafe s hq
      | change Safety.stepSafe (engine (withDot M dot)) bounds (.assign _ _ _ _) s
        simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
          safeBlockPtr2,safeAdvance2]
        simp [Safety.safeOp_unfold]
  · intro st member
    rw [entrySource] at member
    obtain ⟨i,rfl⟩ := List.mem_ofFn.mp member
    fin_cases i
    all_goals change ("Q_block_ptr" : RegName) ≠ _
    all_goals decide

theorem entry_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).before (bindScale scale s) := by
  have split : (parts p).before = preQuery p ++ (parts p).before.drop 9 :=
    (List.take_append_drop 9 _).symm
  rw [split,Safety.traceSafeList_append]
  refine ⟨preQuery_safe M dot p _ bounds,?_⟩
  intro t ht
  exact entry_tail_safe M dot p s t (preQuery_pointer M dot p scale s t ht) bounds hb

#axiomsClean entry_safe
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
