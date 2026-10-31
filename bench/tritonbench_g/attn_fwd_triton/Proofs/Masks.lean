import bench.tritonbench_g.attn_fwd_triton.Proofs.SourceFragments

/-! The candidate's exact causal and tail masks. Inactive scores are bottom,
so the unchanged probability proof applies to the actual repaired source. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Masks
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite center probability)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
noncomputable section
set_option maxHeartbeats 150000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def keep (BM BN N startM startN : Nat) (diagonal : Bool) (row : Fin BM) (key : Fin BN) : Bool :=
  if diagonal then decide (startN+key.val ≤ startM*BM+row.val ∧ startN+key.val < N)
  else decide (startN+key.val < N)

def masked {BM BN : Nat} (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool) : Tile .real [BM,BN] :=
  ⟨fun i => if active i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩

theorem eval_tail (s : BlockState) (BN N startN : Nat)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar startN))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar N)) :
    evalOp (tail BN) s = some (⟨fun i => decide (startN+i.2.1.val < N)⟩ : Tile .bool [1,BN]) := by
  simp only [tail, evalOp.eq_def, hn, hs, hctx, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, NumericDType.sub,
    RealExpressions.tail_mask_iff,ControlView.signed_tail, -Tile.scalar_eta]
  change ((j.val : Int) < (N : Int)-(startN : Int)) ↔ startN+j.val<N
  omega

theorem eval_mask (s : BlockState) (BM BN startM startN : Nat)
    (hm : s.regs .int [BM] "offs_m"=some ⟨fun i => ((startM*BM+i.1.val : Nat) : Int)⟩)
    (hn : s.regs .int [BN] "offs_n"=some (Tile.vec (fun i => (i.val : Int))))
    (hs : s.regs .int [] "start_n"=some (Tile.scalar (startN : Int))) :
    evalOp (maskExpr BM BN) s=some (RealExpressions.maskTile BM BN startM startN) := by
  simp only [maskExpr,evalOp.eq_def,hm,hn,hs,bind,Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,j,⟨⟩⟩
  simp only [Tile.cop_data,Tile.bop_data,Tile.expandDim_data,Tile.vec,Tile.scalar,
    TileShape.dropInsertedIndex,ComparableDType.ge,NumericDType.add,RealExpressions.maskTile]
  simp only [←Int.natCast_add]
  simp [Broadcast.leftIndex,Broadcast.rightIndex]
  simp only [←Int.natCast_mul,←Int.natCast_add]
  change ((startN+j.val : Nat) : Int) ≤ ((startM*BM+r.val : Nat) : Int) ↔
    startN+j.val≤startM*BM+r.val
  omega

theorem guard_executes (s : BlockState) (BM BN N startM startN : Nat) (second diagonal : Bool)
    (score : Fin BM → Fin BN → ℝ)
    (hqk : s.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1)))
    (hm : s.regs .int [BM] "offs_m" = some ⟨fun i => startM*BM+i.1.val⟩)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar startN))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar N)) :
    ∃ t, stepStmts (guard BM BN second diagonal) s = some t ∧
      t.regs .real [BM,BN] "qk" = some (masked score (keep BM BN N startM startN diagonal)) ∧
      Frame [maskName second,"qk"] s t := by
  have hz (st : BlockState) : evalOp (.broadcast .negInf [BM,BN]) st =
      some (⟨fun _ => ⊥⟩ : Tile .real [BM,BN]) := by
    simp only [evalOp.eq_def, bind, Option.bind]
    rfl
  have hname : maskName second ≠ "qk" := by cases second <;> decide
  cases diagonal
  · have ht := eval_tail s BN N startN hn hs hctx
    have he : evalOp (.where (.remap [BM,BN] (Broadcast.leftIndex (.consL (.consSame .nil))) (tail BN))
        (.ref .real [BM,BN] "qk") (.broadcast .negInf [BM,BN])) s =
        some (masked score (keep BM BN N startM startN Bool.false)) := by
      have hr : evalOp (.remap [BM,BN] (Broadcast.leftIndex (.consL (.consSame .nil))) (tail BN)) s =
          some (⟨fun i => decide (startN+i.2.1.val < N)⟩ : Tile .bool [BM,BN]) := by
        rw [evalOp.eq_def]
        simp only [ht, bind, Option.bind]
        rfl
      rw [evalOp_where]
      simp only [hr, evalOp_ref, hqk, hz, bind, Option.bind]
      congr 1
    simp only [guard, Bool.false_eq_true, ↓reduceIte]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some he), stepStmts.nil]
    exact ⟨_, rfl, by simp, Frame.setReg _ s "qk" _ _ _ (by simp)⟩
  · let sm := s.setReg (maskName second) .bool [BM,BN] (RealExpressions.maskTile BM BN startM startN)
    have hmask := eval_mask s BM BN startM startN hm hn hs
    have ht := eval_tail sm BN N startN (by cases second <;> simpa [sm,maskName,RealPreparation.maskName] using hn)
      (by cases second <;> simpa [sm,maskName,RealPreparation.maskName] using hs)
      (by cases second <;> simpa [sm,maskName,RealPreparation.maskName] using hctx)
    have hqk' : sm.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1)) := by
      simpa [sm, Ne.symm hname] using hqk
    have he : evalOp (.where (.boolAnd (.consR (.consSame .nil))
        (.ref .bool [BM,BN] (maskName second)) (tail BN))
        (.ref .real [BM,BN] "qk") (.broadcast .negInf [BM,BN])) sm =
        some (masked score (keep BM BN N startM startN Bool.true)) := by
      have hmref : sm.regs .bool [BM,BN] (maskName second) =
          some (RealExpressions.maskTile BM BN startM startN) := by simp [sm]
      have ha : evalOp (.boolAnd (.consR (.consSame .nil))
          (.ref .bool [BM,BN] (maskName second)) (tail BN)) sm =
          some (⟨fun i => decide (startN+i.2.1.val ≤ startM*BM+i.1.val) &&
            decide (startN+i.2.1.val < N)⟩ : Tile .bool [BM,BN]) := by
        rw [evalOp.eq_def]
        simp only [evalOp_ref, hmref, ht, bind, Option.bind]
        rfl
      rw [evalOp_where]
      simp only [ha, evalOp_ref, hqk', hz, bind, Option.bind]
      congr 1
      apply Tile.ext
      rintro ⟨r,j,⟨⟩⟩
      simp [masked, keep, finite, RealExpressions.maskTile, Tile.select_data, Tile.bop_data,
        Tile.scalar, -Tile.scalar_eta]
      rfl
    simp only [guard, Bool.true_eq, ↓reduceIte]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some hmask),
      stepStmts.cons_some (stepStmt_assign_eq_some he), stepStmts.nil]
    exact ⟨_, rfl, by simp, (Frame.setReg _ s (maskName second) _ _ _ (by simp)).trans
      (Frame.setReg _ sm "qk" _ _ _ (by simp))⟩

theorem probabilities_execute (s : BlockState) (BM BN : Nat) (hN : 0 < BN)
    (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hqk : s.regs .real [BM,BN] "qk" = some (masked score active)) :
    ∃ t, stepStmts (RealBody.center BM BN ++ [RealBody.probability BM BN]) s = some t ∧
      t.regs .real [BM] "m_ij" = some (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) ∧
      t.regs .real [BM,BN] "p" = some (finite (fun i : TileIndex [BM,BN] => probability score active old i.1 i.2.1)) ∧
      Frame ["m_ij","qk","p"] s t := by
  exact VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealProbabilities.prefix_executes
    s BM BN 0 0 0 hN score active old hm hqk

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Masks
