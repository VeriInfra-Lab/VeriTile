import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.IOPhaseSafety

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO.SourceSafety
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody SharedStep
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem invariant_list [Inhabited α] (code : List ComputeStmt) (bounds : RegionName → Nat)
    (I : State (Term α) → Prop)
    (safe : ∀ st∈code, ∀ s, I s → Safety.stepSafe algebra bounds st s)
    (preserved : ∀ st∈code, ∀ s t, I s → step algebra st s=some t → I t)
    (s : State (Term α)) (initial : I s) : Safety.traceSafeList algebra bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s initial)
    intro t ran
    exact ih (fun st h => safe st (by simp [h]))
      (fun st h => preserved st (by simp [h])) t (preserved st (by simp) s t initial ran)

structure InputPointers (c : Shape) (head chunk : Nat) (s : State (Term α)) : Prop where
  q : s.regs .blockPtr [32,c.D] "p_q"=some (fun _=>mainPtr c head chunk "q")
  k : s.regs .blockPtr [c.D,32] "p_k"=some (fun _=>keyPtr c head chunk)
  v : s.regs .blockPtr [32,c.D] "p_v"=some (fun _=>mainPtr c head chunk "v")

theorem assign_pointers [Inhabited α] (c : Shape) (head chunk : Nat)
    (d : TileDType) (sh : TileShape) (name : RegName) (e : ComputeExpr d sh)
    (hq : "p_q"≠name) (hk : "p_k"≠name) (hv : "p_v"≠name)
    (s t : State (Term α)) (initial : InputPointers c head chunk s)
    (ran : step algebra (.assign d sh name e) s=some t) : InputPointers c head chunk t := by
  simp only [step] at ran
  cases value : evalExpr algebra e s with
  | none => simp [value] at ran
  | some v =>
    simp [value] at ran
    subst t
    rcases initial with ⟨q,k,v⟩
    constructor <;> simpa only [State.setReg_ne_name _ _ _ _ _ _ _ _ hq,
      State.setReg_ne_name _ _ _ _ _ _ _ _ hk,State.setReg_ne_name _ _ _ _ _ _ _ _ hv]

theorem load_safe [Inhabited α] (m n : Nat) (pointer value : String) (ptr : BlockPtr)
    (axes : List Nat) (bounds : RegionName → Nat) (s : State (Term α))
    (hp : s.regs .blockPtr [m,n] pointer=some (fun _=>ptr))
    (bounded : ∀ i : TileIndex [m,n],ptr.inBounds (TileShape.indexToList [m,n] i) axes=Bool.true →
      ptr.address (TileShape.indexToList [m,n] i)<bounds ptr.region) :
    Safety.stepSafe algebra bounds (Program.load m n pointer value axes) s := by
  simpa [Program.load,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,hp] using bounded

theorem loads_safe [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat) (storeFlag : Bool)
    (carry : Matrix α c.D c.D) (bounds : RegionName → Nat) (bounded : Bounds c storeFlag bounds)
    (headBound : head<c.heads) (s : State (Term α)) :
    Safety.traceSafeList algebra bounds (Program.loads c scale) (saved c head chunk storeFlag carry s) := by
  apply invariant_list _ bounds (InputPointers c head chunk)
  · intro st member t ptrs
    simp only [Program.loads,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl
    · apply load_safe _ _ _ _ (mainPtr c head chunk "q") _ _ _ ptrs.q
      intro i active
      change (mainPtr c head chunk "q").inBounds [i.1.val,i.2.1.val] [0]=Bool.true at active
      rw [main_bound,decide_eq_true_eq] at active
      change (mainPtr c head chunk "q").address [i.1.val,i.2.1.val]<bounds "q"
      rw [main_address]
      exact bounded.main "q" (by simp) _ (c.main_bound head (chunk*32+i.1.val) i.2.1.val headBound active i.2.1.isLt)
    · simp only [Program.put,Program.comp,Program.cast,Program.ref,Safety.stepSafe_unfold,
        Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]
    · apply load_safe _ _ _ _ (keyPtr c head chunk) _ _ _ ptrs.k
      intro i active
      change (keyPtr c head chunk).inBounds [i.1.val,i.2.1.val] [1]=Bool.true at active
      rw [key_bound,decide_eq_true_eq] at active
      change (keyPtr c head chunk).address [i.1.val,i.2.1.val]<bounds "k"
      rw [key_address]
      exact bounded.main "k" (by simp) _ (c.main_bound head (chunk*32+i.2.1.val) i.1.val headBound active i.1.isLt)
    · apply load_safe _ _ _ _ (mainPtr c head chunk "v") _ _ _ ptrs.v
      intro i active
      change (mainPtr c head chunk "v").inBounds [i.1.val,i.2.1.val] [0]=Bool.true at active
      rw [main_bound,decide_eq_true_eq] at active
      change (mainPtr c head chunk "v").address [i.1.val,i.2.1.val]<bounds "v"
      rw [main_address]
      exact bounded.main "v" (by simp) _ (c.main_bound head (chunk*32+i.1.val) i.2.1.val headBound active i.2.1.isLt)
    all_goals simp only [Program.put,Program.comp,Program.cast,Program.ref,Program.dot,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]
  · intro st member t u ptrs ran
    simp only [Program.loads,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl
    all_goals exact assign_pointers c head chunk _ _ _ _ (by decide) (by decide) (by decide) t u ptrs ran
  · constructor <;> simp [saved_regs,pointers]

theorem later_safe [Inhabited α] (c : Shape) (bounds : RegionName → Nat) (s : State (Term α)) :
    Safety.traceSafeList algebra bounds (Program.later c) s := by
  apply pure_list
  intro st member t
  simp only [Program.later,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl
  all_goals simp only [Program.put,Program.comp,Program.cast,Program.ref,Program.dot,Program.add,Program.carryDot,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]

theorem update_safe [Inhabited α] (c : Shape) (ifcond : Bool)
    (bounds : RegionName → Nat) (s : State (Term α)) :
    Safety.stepSafe algebra bounds (Program.update c ifcond) s := by
  have carrySafe : Safety.traceSafeList algebra bounds [Program.put "b_h" (Program.carryDot c)] s := by
    apply pure_list
    intro st member t
    simp only [List.mem_singleton] at member
    subst st
    simp only [Program.put,Program.comp,Program.ref,Program.dot,Program.carryDot,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]
  have inner : Safety.stepSafe algebra bounds
      (.ifThenElse (.eq .nat .nil (.ref .nat [] "i") (.constNat 0))
        [Program.put "b_h" (Program.carryDot c)] (Program.later c)) s := by
    simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,true_and]
    cases h : evalOp algebra none (.eq .nat .nil (.ref .nat [] "i") (.constNat 0)) s with
    | none => simp only [h]
    | some b =>
      simp only [h]
      split <;> first | exact carrySafe | exact later_safe c bounds s
  cases ifcond <;> simp only [Program.update,Safety.stepSafe_unfold,Safety.safeOp_unfold,
    evalOp_unfold,Bool.false_eq_true,if_false,if_true,true_and]
  · exact later_safe c bounds s
  · exact Safety.traceSafeList_cons_intro inner (fun t _=>Safety.traceSafeList_nil _ _ _)

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO.SourceSafety
