import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoop
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealRegisterSafety

/-! Access safety of one complete query iteration. The source's selected load
mode determines active Q lanes, while M lanes are active only below N.
Arithmetic, window selection and pointer advancement do not access memory. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyParts
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

def loadChoice (shape : TileShape) (pointer target : RegName) (even : Bool) : Stmt :=
  .ifThenElse (.constBool even)
    [.assign .real shape target (.load .real (.blockPtr (.ref .blockPtr shape pointer) []) .none .unspecified)]
    [.assign .real shape target (.load .real (.blockPtr (.ref .blockPtr shape pointer) [0,1]) .none)]

theorem load_choice_code (shape : TileShape) (pointer target : RegName) (even : Bool) :
    RealFrame.Code [target] [loadChoice shape pointer target even] := by
  unfold loadChoice
  repeat' first
    | apply RealFrame.Code.ifThenElse
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.nil
    | simp

theorem load_choice_safe (bounds : RegionBounds) (s : BlockState) (shape : TileShape)
    (pointer target : RegName) (even : Bool) (ptrs : Tile .blockPtr shape)
    (hp : s.regs .blockPtr shape pointer = some ptrs)
    (hb : BlockBounds bounds ptrs (RealLoads.checks even)) :
    Stmt.TraceSafe bounds (loadChoice shape pointer target even) s := by
  cases even
  all_goals
    simp only [loadChoice, Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def,
      Tile.scalar_data, Bool.false_eq_true, ↓reduceIte, true_and]
    apply Stmt.TraceSafeList.cons_intro
    · simpa only [Stmt.TraceSafe, Op.SafeAt.eq_def] using block_load_safe bounds s pointer ptrs _ hp hb
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem arithmetic_safe (B D N : Nat) (offset size : Int)
    (window complement evenN candidate : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealArithmetic.body B D N offset size window complement evenN candidate) s := by
  apply list_safe
  cases candidate <;> simp [RealArithmetic.body, RealExpressions.distanceExpr, Op.SafeAt.eq_def]

theorem suffix_safe (B D : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.suffix B D) s := by
  apply list_safe
  simp [RealBody.suffix, Op.SafeAt.eq_def]

theorem prefix_safe (bounds : RegionBounds) (s : BlockState) (Q M : RegionName)
    (B D N baseQ baseM sqm sqk row : Nat) (evenM : Bool)
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk row))
    (hm : s.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM row))
    (hbQ : BlockBounds bounds (RealLoads.queryPointer Q B D N baseQ sqm sqk row) (RealLoads.checks evenM))
    (hr : s.regs .nat [] "start_m" = some (Tile.scalar row))
    (hbM : ∀ i : Fin B, row+i.val < N → baseM+row+i.val < bounds M) :
    Stmt.TraceSafeList bounds (RealBody.loadPrefix B D N evenM) s := by
  unfold RealBody.loadPrefix
  apply TraceProofs.assign_eval_cons (by simp [Op.SafeAt.eq_def])
  intro index hindex
  have hi : index = Tile.scalar row := by simpa [evalOp_ref,hr] using hindex.symm
  subst index
  apply Stmt.TraceSafeList.cons_intro
    (load_choice_safe bounds _ [B,D] "Q_block_ptr" "q" evenM _ (by simp [hp]) hbQ)
  intro t ht
  have he : stepStmts [loadChoice [B,D] "Q_block_ptr" "q" evenM]
      (s.setReg "start_m" .nat [] (Tile.scalar row)) = some t := by
    simpa only [stepStmts, ht, bind, Option.bind] using (rfl : some t = some t)
  have frame := (load_choice_code [B,D] "Q_block_ptr" "q" evenM).frame he
  have hmt : t.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM row) :=
    (frame.regs .ptr [B] "m_ptrs" (by simp)).trans (by simp [hm])
  have hrt : t.regs .nat [] "start_m" = some (Tile.scalar row) :=
    (frame.regs .nat [] "start_m" (by simp)).trans (by simp)
  apply Stmt.TraceSafeList.cons_intro
  · simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, true_and]
    unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
    refine ⟨by simp [Op.MemorySafe], ?_⟩
    intro tile ht i hactive
    rw [evalOp_ref, hmt] at ht
    cases ht
    have hrow : row + i.1.val < N := by
      simpa [MaskOpt.Active, evalOp.eq_def, hrt, bind, Option.bind,
        Tile.bop, Tile.cop, Tile.scalar, Tile.vec, NumericDType.add,
        ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta] using hactive
    simpa [RealLoads.normalizerPointers] using hbM i.1 hrow
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem body_safe (bounds : RegionBounds) (s : BlockState) (Q M : RegionName)
    (B D N baseQ baseM sqm sqk row : Nat) (offset size : Int)
    (window complement evenM evenN candidate : Bool)
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk row))
    (hm : s.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM row))
    (hbQ : BlockBounds bounds (RealLoads.queryPointer Q B D N baseQ sqm sqk row) (RealLoads.checks evenM))
    (hr : s.regs .nat [] "start_m" = some (Tile.scalar row))
    (hbM : ∀ i : Fin B, row+i.val < N → baseM+row+i.val < bounds M) :
    Stmt.TraceSafeList bounds (RealBody.body B D N offset size window complement evenM evenN candidate) s := by
  unfold RealBody.body
  apply append_safe
  · apply append_safe _ s (prefix_safe bounds s Q M B D N baseQ baseM sqm sqk row evenM hp hm hbQ hr hbM)
    intro t _
    exact arithmetic_safe B D N offset size window complement evenM candidate bounds t
  · intro t _
    exact suffix_safe B D bounds t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSafetyParts
