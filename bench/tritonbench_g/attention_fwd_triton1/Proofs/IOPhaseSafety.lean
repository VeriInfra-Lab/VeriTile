import bench.tritonbench_g.attention_fwd_triton1.IO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO.SourceSafety
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody SharedStep
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure Bounds (c : Shape) (storeFlag : Bool) (bounds : RegionName → Nat) : Prop where
  main : ∀ r∈["q","k","v","o"],∀ off,off<c.mainSize → off<bounds r
  history : storeFlag=Bool.true → ∀ off,off<c.historySize → off<bounds "h"

theorem pure_list [Inhabited α] (code : List ComputeStmt) (bounds : RegionName → Nat)
    (safe : ∀ st∈code, ∀ s : State (Term α), Safety.stepSafe algebra bounds st s)
    (s : State (Term α)) : Safety.traceSafeList algebra bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h => safe st (by simp [h])) t

theorem pointer_safe [Inhabited α] (name region : String) (parent block strides : List Nat)
    (base : Op .nat []) (offsets : List (Op .nat [])) (bounds : RegionName → Nat)
    (s : State (Term α)) (baseSafe : Safety.safeOp algebra none bounds s base)
    (offsetSafe : ∀ off∈offsets, Safety.safeOp algebra none bounds s off) :
    Safety.stepSafe algebra bounds (Program.pointer name region parent block strides base offsets) s := by
  simp only [Program.pointer,Safety.stepSafe_unfold,Safety.safeExpr_unfold]
  rw [Safety.safeOp_unfold]
  exact ⟨baseSafe,offsetSafe⟩

theorem pointers_safe [Inhabited α] (c : Shape)
    (bounds : RegionName → Nat) (s : State (Term α)) :
    Safety.traceSafeList algebra bounds (Program.pointers c) s := by
  apply pure_list
  intro st member t
  simp only [Program.pointers,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl
  all_goals apply pointer_safe
  all_goals first
    | (intro off member
       simp only [List.mem_cons,List.not_mem_nil,or_false] at member
       rcases member with rfl|rfl
       all_goals simp only [Program.times,Program.headTimes,Safety.safeOp_unfold,and_self])
    | simp only [Program.times,Program.headTimes,Safety.safeOp_unfold,and_self]

theorem store_safe [Inhabited α] (m n : Nat) (pointer value : String) (ptr : BlockPtr)
    (axes : List Nat) (bounds : RegionName → Nat) (s : State (Term α))
    (hp : s.regs .blockPtr [m,n] pointer=some (fun _=>ptr))
    (bounded : ∀ i : TileIndex [m,n],ptr.inBounds (TileShape.indexToList [m,n] i) axes=Bool.true →
      ptr.address (TileShape.indexToList [m,n] i)<bounds ptr.region) :
    Safety.stepSafe algebra bounds (Program.store m n pointer value axes) s := by
  simpa [Program.store,Program.comp,Program.cast,Program.ref,Safety.stepSafe,Safety.safeExpr,
    Safety.safeCompute,Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,
    Safety.maskActive,evalOp_unfold,hp] using bounded

theorem save_safe [Inhabited α] (c : Shape) (head chunk : Nat) (storeFlag : Bool)
    (bounds : RegionName → Nat) (bounded : Bounds c storeFlag bounds)
    (headBound : head<c.heads) (chunkBound : chunk<c.chunks) (s : State (Term α)) :
    Safety.traceSafeList algebra bounds [Program.save c storeFlag] (pointers c head chunk s) := by
  apply Safety.traceSafeList_cons_intro
  · simp only [Program.save,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,true_and]
    cases hs : storeFlag
    · trivial
    · change Safety.traceSafeList algebra bounds [Program.store c.D c.D "p_h" "b_h" []] _
      apply Safety.traceSafeList_cons_intro
      · apply store_safe _ _ _ _ (historyPtr c head chunk) [] bounds _ (by simp [pointers])
        intro i _
        apply bounded.history hs
        change (historyPtr c head chunk).address [i.1.val,i.2.1.val]<c.historySize
        rw [history_address]
        exact c.history_bound head chunk i.1 i.2.1 headBound chunkBound i.1.isLt i.2.1.isLt
      · intro final _; exact Safety.traceSafeList_nil _ _ _
  · intro final _; exact Safety.traceSafeList_nil _ _ _

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO.SourceSafety
