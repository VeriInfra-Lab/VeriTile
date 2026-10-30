import bench.repairs.tritonbench_g.attention_llama.Proofs.FP8Loop
import bench.tritonbench_g.attention_llama.Proofs.FP8RealSafety

/-! Repaired byte-source access safety. Reuses the original masked-load,
setup, cursor, bounds and postamble interfaces; private source-structural
lemmas are specialized to the moved global score mask. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.ByteSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open RealKernel FP8RealSource FP8RealLoop FP8RealLoopFrame
open FP8RealSafetyLoop (AccessesFrom)
open FP8Iteration
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

private theorem lowerLoad_fuse (st : Stmt) (s : BlockState) :
    stepStmts (lowerLoad Bool.false st) s = stepStmts (lowerLoad Bool.true st) s := by
  simpa using FP8Source.lowerList_fuse [st] s

private theorem ref_safe (bounds : RegionBounds) (s : BlockState) (d : TileDType)
    (sh : TileShape) (name : RegName) :
    (Op.ref d sh name).SafeAt bounds s = True := by rw [Op.SafeAt.eq_def]

private theorem decode_safe (bounds : RegionBounds) (s : BlockState)
    (bytes : Op .int shape) :
    @Op.SafeAt .real shape bounds s (.decodeFloat8E5 .real bytes) =
      bytes.SafeAt bounds s := by rw [Op.SafeAt.eq_def]

private theorem conversion_safe (bounds : RegionBounds) (name : RegName)
    (bytes : Op .int shape) (s : BlockState) :
    Stmt.TraceSafeList bounds (FP8RealLoads.conversion name bytes) s ↔
      bytes.SafeAt bounds s := by
  simp only [FP8RealLoads.conversion, Stmt.TraceSafeList, Stmt.TraceSafe,
    decode_safe, ref_safe, true_and, and_true]
  constructor
  · exact And.left
  · intro h
    refine ⟨h, ?_⟩
    split <;> try trivial
    split <;> try trivial
    split <;> trivial

private theorem lowerLoad_safe (bounds : RegionBounds) (st : Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (lowerLoad Bool.false st) s ↔
      Stmt.TraceSafeList bounds (lowerLoad Bool.true st) s := by
  cases h : byteLoad? st with
  | none => simp [lowerLoad, h]
  | some load =>
    simp only [lowerLoad, h, Bool.false_eq_true, ↓reduceIte, conversion_safe,
      Stmt.TraceSafeList, Stmt.TraceSafe, decode_safe, and_true]
    cases stepStmt (.assign .real load.shape load.name
      (.decodeFloat8E5 .real load.bytes)) s <;> simp

private theorem lowerList_safe (bounds : RegionBounds) (code : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (code.flatMap (lowerLoad Bool.false)) s ↔
      Stmt.TraceSafeList bounds (code.flatMap (lowerLoad Bool.true)) s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [List.flatMap_cons, TraceRewrite.append_iff, lowerLoad_safe, lowerLoad_fuse, ih]

private theorem max_head_safe {n : Nat} (bounds : RegionBounds) (x y : Op .real [n])
    (name : RegName) (rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.assign .real [n] name (.max2 (.consSame .nil) x y) :: rest) s ↔
      Stmt.TraceSafeList bounds (.assign .real [n] name (.maximumFloat .real (.consSame .nil) x y) :: rest) s := by
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, stepStmt, FloatDType.toTileDType]
  erw [evalOp_maximumFloat_real]
  conv_lhs => arg 1; rw [Op.SafeAt.eq_def]
  conv_rhs => arg 1; rw [Op.SafeAt.eq_def]
  rfl

private theorem source_tail_safe (bounds : RegionBounds) (skn svk N BM BN BD : Nat)
    (s : BlockState) :
    Stmt.TraceSafeList bounds
      ((FP8RealSource.sourceTail skn svk N BM BN BD Bool.false).flatMap (FP8RealSource.lowerLoad Bool.true)) s ↔
    Stmt.TraceSafeList bounds
      ((alLoopTailG skn svk N BM BN BD Bool.false).flatMap (FP8RealSource.lowerLoad Bool.true)) s := by
  exact max_head_safe bounds _ _ "m_curr"
    (((alLoopTailG skn svk N BM BN BD Bool.false).drop 1).flatMap (FP8RealSource.lowerLoad Bool.true)) s

private theorem inline_safe (bounds : RegionBounds) (scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (causal : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds
      (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s ↔
    Stmt.TraceSafeList bounds (inlineLoop scale skn svk N SP BM BN BD causal) s := by
  simp only [FP8Source.loopBody, inlineLoop, List.append_assoc, TraceRewrite.append_iff,
    TraceRewrite.constant_branch, source_tail_safe]

private theorem load_safe (bounds : RegionBounds) (R : RegionName) (s : BlockState)
    (name : RegName) (base sr sc SN N BN BD : Nat)
    (hptr : s.regs .ptr [BN, BD] name = some (alPtrTile R base sr sc SN BN BD))
    (hrows : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hb : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      base + (SN + i.1.val) * sr + i.2.1.val * sc < bounds R) :
    Op.SafeAt bounds s (.decodeFloat8E5 .real
      (.load .int (.ptr (.ref .ptr [BN, BD] name))
        (.maskOther (keyMask N BN BD) (.broadcast (.constInt 0) [BN, BD])))) := by
  simp only [Op.SafeAt.eq_def, keyMask, true_and, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def,
    MaskOpt.Active]
  intro ptrs hp i hi
  rw [evalOp_ref, hptr] at hp
  cases hp
  rcases hi with ⟨mask, hm, hi⟩
  rw [al_colmask_eval s SN N hrows] at hm
  cases hm
  exact hb i (of_decide_eq_true hi)

/-- Both complete iterations access only the indicated masked K/V addresses.
No byte value or numerical accumulator equation is a premise. -/
theorem iteration_safe (bounds : RegionBounds) (K V : RegionName) (s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn SN N SP BM BN BD : Nat) (causal : Bool)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoff : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hkp : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hvp : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD))
    (hK : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      kBase + (SN + i.1.val) * skn + i.2.1.val * skk < bounds K)
    (hV : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      vBase + (SN + i.1.val) * svk + i.2.1.val * svn < bounds V) :
    Stmt.TraceSafeList bounds
      (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s := by
  have hoffEval : evalOp (.add .nat Broadcast.scalarL (.ref .nat [] "start_n")
      (.ref .nat [BN] "offs_n")) s = some (Tile.vec (fun j : Fin BN => SN + j.val)) := by
    rw [evalOp_add]
    simp only [evalOp_ref, hsn, hoff, Option.bind_eq_bind, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    intro i
    rfl
  rw [inline_safe]
  cases causal
  all_goals
    simp only [inlineLoop, SourceExecution.loopPrefix, alLoopTailG, FP8RealSource.lowerLoad,
      FP8RealSource.byteLoad?, List.flatMap_cons, List.flatMap_nil, List.cons_append,
      List.nil_append, Bool.false_eq_true, ↓reduceIte, Bool.or_eq_true, beq_iff_eq,
      String.reduceEq, false_or, or_false, show (0.0 : ℝ) = 0 from by norm_num]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hoffEval)]
    refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
    apply TraceProofs.assign_cons
    · exact load_safe bounds K _ "k_ptrs" kBase skn skk SN N BN BD
        (by simpa using hkp) (by simp) hK
    · intro kt
      repeat' (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
      apply TraceProofs.assign_cons
      · exact load_safe bounds V _ "v_ptrs" vBase svk svn SN N BN BD
          (by simpa using hvp) (by simp) hV
      · intro vt
        apply TraceProofs.all_safe
        simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

private def assignsAway (target : RegName) : Stmt → Prop
  | .assign _ _ name _ => name ≠ target
  | _ => False

private theorem assignments_frame (name : RegName) (d : TileDType) (sh : TileShape)
    (code : List Stmt) (hcode : ∀ st ∈ code, assignsAway name st)
    (s t : BlockState) (hrun : stepStmts code s = some t) :
    t.regs d sh name = s.regs d sh name := by
  induction code generalizing s with
  | nil =>
    simp only [stepStmts.nil, Option.some.injEq] at hrun
    subst t
    rfl
  | cons st rest ih =>
    have hc := hcode st (by simp)
    have hrest : ∀ st ∈ rest, assignsAway name st :=
      fun st h => hcode st (by simp [h])
    cases st <;> simp only [assignsAway] at hc
    rename_i dtype shape dest e
    cases he : evalOp e s with
    | none => simp [stepStmts, stepStmt, he] at hrun
    | some tile =>
      have hr : stepStmts rest (s.setReg dest dtype shape tile) = some t := by
        simpa only [stepStmts, stepStmt, he, bind, Option.bind_some] using hrun
      rw [ih hrest _ hr]
      simp [Ne.symm hc]

private theorem assignments_pids (code : List Stmt)
    (hcode : ∀ st ∈ code, ∃ d sh name e, st = .assign d sh name e)
    (s t : BlockState) (hrun : stepStmts code s = some t) : t.pids = s.pids := by
  induction code generalizing s with
  | nil =>
    simp only [stepStmts.nil, Option.some.injEq] at hrun
    subst t
    rfl
  | cons st rest ih =>
    obtain ⟨d, sh, name, e, rfl⟩ := hcode st (by simp)
    cases he : evalOp e s with
    | none => simp [stepStmts, stepStmt, he] at hrun
    | some tile =>
      have hr : stepStmts rest (s.setReg name d sh tile) = some t := by
        simpa only [stepStmts, stepStmt, he, bind, Option.bind_some] using hrun
      exact ih (fun st h => hcode st (by simp [h])) (s.setReg name d sh tile) hr

/-- Both complete loop variants retain the program IDs and batch/head
registers needed for the final output store. Numerical values are unrestricted. -/
theorem launch_preserved (initial s t : BlockState) (NH : Nat) (scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (causal : Bool)
    (hl : Launch initial NH s)
    (hrun : stepStmts (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s = some t) :
    Launch initial NH t := by
  rw [source_inline] at hrun
  have hcode (name : RegName) (hn : name = "batch_id" ∨ name = "off_hz") :
      ∀ st ∈ inlineLoop scale skn svk N SP BM BN BD causal, assignsAway name st := by
    rcases hn with rfl | rfl
    all_goals cases causal <;>
      simp [inlineLoop, SourceExecution.loopPrefix, alLoopTailG, alCausalWhereStmt,
        FP8RealSource.lowerLoad, FP8RealSource.byteLoad?, assignsAway]
  refine ⟨?_, ?_, ?_⟩
  · apply Eq.trans (assignments_pids _ ?_ s t hrun) hl.pids
    intro st hs
    have h := hcode "batch_id" (Or.inl rfl) st hs
    cases st <;> simp only [assignsAway] at h
    exact ⟨_, _, _, _, rfl⟩
  · exact (assignments_frame "batch_id" .nat [] _ (hcode _ (Or.inl rfl)) s t hrun).trans hl.batch
  · exact (assignments_frame "off_hz" .nat [] _ (hcode _ (Or.inr rfl)) s t hrun).trans hl.head

private theorem accesses_next (bounds : RegionBounds) (R : RegionName)
    (base sr sc cur stop N BN BD : Nat)
    (h : AccessesFrom bounds R base sr sc cur stop N BN BD) :
    AccessesFrom bounds R base sr sc (cur + BN) stop N BN BD := by
  intro k hk j d hj
  have he : cur + BN + k * BN = cur + (k + 1) * BN := by
    simp only [Nat.add_mul, Nat.one_mul]
    omega
  rw [he] at hk hj ⊢
  exact h (k + 1) hk j d hj

/-- Bounds at every reached iteration, including partial key tails, zero
stride and evaluations that fail before the next iteration. -/
theorem range_safe (bounds : RegionBounds) (K V : RegionName) (initial s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn cur stop N SP BM BN BD : Nat) (causal : Bool)
    (hc : Cursor K V initial kBase vBase skn skk svk svn BN BD cur s)
    (hK : AccessesFrom bounds K kBase skn skk cur stop N BN BD)
    (hV : AccessesFrom bounds V vBase svk svn cur stop N BN BD) :
    Stmt.forRangeTraceSafe bounds "start_n" cur stop BN
      (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s := by
  rw [Stmt.forRangeTraceSafe]
  by_cases hz : BN = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hi : cur < stop
    · simp only [hi, ↓reduceIte]
      refine ⟨?_, ?_⟩
      · apply iteration_safe bounds K V _ scale kBase vBase skn skk svk svn cur N SP BM BN BD
          causal (by simp) (by simpa using hc.offsets)
          (by simpa using hc.keys) (by simpa using hc.values)
        · intro i hj
          simpa using hK 0 (by simpa using hi) i.1 i.2.1 (by simpa using hj)
        · intro i hj
          simpa using hV 0 (by simpa using hi) i.1 i.2.1 (by simpa using hj)
      · cases he : stepStmts
            (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal)
            (s.setReg "start_n" .nat [] (Tile.scalar cur)) with
        | none => trivial
        | some t =>
          exact range_safe bounds K V initial t scale kBase vBase skn skk svk svn
            (cur + BN) stop N SP BM BN BD causal
            (FP8Loop.cursor_next K V initial s t scale kBase vBase skn skk svk svn cur N SP BM BN BD causal hc he)
            (accesses_next bounds K kBase skn skk cur stop N BN BD hK)
            (accesses_next bounds V vBase svk svn cur stop N BN BD hV)
    · simp only [hi, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

/-- A successful complete range retains the launch metadata used by Out. -/
theorem range_launch (initial s t : BlockState) (NH : Nat) (scale : ℝ)
    (skn svk cur stop N SP BM BN BD : Nat) (causal : Bool)
    (hl : Launch initial NH s)
    (hrun : stepForRangeAux "start_n" cur stop BN
      (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s = some t) :
    Launch initial NH t := by
  by_cases hz : BN = 0
  · subst BN
    rw [stepForRangeAux.step_zero_step, Option.some.injEq] at hrun
    subst t
    exact hl
  · by_cases hi : cur < stop
    · rw [stepForRangeAux.step_lt hz hi] at hrun
      cases he : stepStmts
          (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal)
          (s.setReg "start_n" .nat [] (Tile.scalar cur)) with
      | none => simp [he] at hrun
      | some mid =>
        simp only [he, Option.bind_some] at hrun
        apply range_launch initial mid t NH scale skn svk (cur + BN) stop N SP BM BN BD causal
          (hrun := hrun)
        apply launch_preserved initial _ mid NH scale skn svk N SP BM BN BD causal (hrun := he)
        exact ⟨hl.pids, by simpa using hl.batch, by simpa using hl.head⟩
    · rw [stepForRangeAux.step_ge hz (by omega), Option.some.injEq] at hrun
      subst t
      exact hl
termination_by stop - cur
decreasing_by omega

private theorem loop_safe (bounds : RegionBounds) (scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (causal : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds (FP8Source.loopBody Bool.false scale skn svk N SP BM BN BD causal) s ↔
      Stmt.TraceSafeList bounds (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s := by
  simp only [FP8Source.loopBody, TraceRewrite.append_iff, lowerList_safe,
    FP8Source.lowerList_fuse, Stmt.TraceSafeList]

private theorem fused_safety
    (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool) (bounds : RegionBounds) (s : BlockState) :
    (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.TraceSafe bounds s ↔
    Stmt.TraceSafeList bounds (FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true) s := by
  simp only [Kernel.TraceSafe, FP8Source.source_body, FP8Source.body, TraceRewrite.append_iff,
    Stmt.TraceSafeList, TraceRewrite.dynamic_range_congr bounds _ _ _ _ _ _
      (FP8Source.loop_fuse scale skn svk N SP BM BN BD causal)
      (loop_safe bounds scale skn svk N SP BM BN BD causal),
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (FP8Source.loop_fuse scale skn svk N SP BM BN BD causal)]

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)

/-- Safety of the actual erased byte source, including all intermediate
loads and the output store. Finite-byte assumptions and numerical equations
are unnecessary: a failed decode cannot execute later accesses. -/
theorem source_safe (bounds : RegionBounds) (s : BlockState)
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < bounds Q)
    (hK : AccessesFrom bounds K (alBase s NH skz skh) skn skk 0
      (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (hV : AccessesFrom bounds V (alBase s NH svz svh) svk svn 0
      (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < bounds Out) :
    (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨entry, he, hc, hb, hl⟩ := FP8RealSetup.runs_with_launch Q K V sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N SP BM BN BD causal s hu
  have hsetup : Stmt.TraceSafeList bounds (FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn NH H N SP BM BN BD causal) s := by
    unfold FP8RealSetup.body
    apply TraceProofs.append_safe
    · exact RealSafetyPrologue.preamble_safe bounds s Q K V sqz sqh sqm sqk skz skh skn skk
        svz svh svk svn NH H N BM BN BD hQ
    · intro t _
      rw [TraceRewrite.constant_branch]
      apply TraceProofs.all_safe
      cases causal <;> simp [Stmt.TraceSafe, Op.SafeAt.eq_def, alBnEndCausalStmt]
  have hloop : Stmt.TraceSafe bounds
      (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal)) entry := by
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp_constNat, evalOp_ref, hb,
      Tile.scalar_data, true_and]
    exact range_safe bounds K V s entry scale (alBase s NH skz skh) (alBase s NH svz svh)
      skn skk svk svn 0 (FP8RealSetup.stop s N SP BN causal) N SP BM BN BD causal hc hK hV
  apply (fused_safety Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD causal bounds s).mpr
  change Stmt.TraceSafeList bounds (FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true) s
  have hbody : FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true =
      FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD causal ++
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) ::
        alPostLoopG Out soz soh som son H BM BD := by
    simp [FP8Source.body, FP8RealSetup.body, List.append_assoc]
  rw [hbody]
  apply TraceProofs.append_safe _ s hsetup
  intro entry' he'
  rw [he, Option.some.injEq] at he'
  subst entry'
  apply Stmt.TraceSafeList.cons_intro hloop
  intro mid hm
  have hr : stepForRangeAux "start_n" 0 (FP8RealSetup.stop s N SP BN causal) BN
      (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) entry = some mid := by
    simpa only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat, evalOp_ref, hb,
      bind, Option.bind_some, Tile.scalar_data] using hm
  have hf := range_launch s entry mid NH scale skn svk 0 (FP8RealSetup.stop s N SP BN causal)
    N SP BM BN BD causal hl hr
  exact RealSafetyPrologue.postamble_safe bounds Out s soz soh som son NH H BM BD mid
    hf.pids hf.batch hf.head hO

end


end VeriTile.Bench.Repairs.AttentionLlama.ByteSafety
