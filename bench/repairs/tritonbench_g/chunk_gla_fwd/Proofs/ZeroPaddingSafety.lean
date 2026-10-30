import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.ZeroPadding
import VeriTile.Meta.StatementAudit

/-! The repaired block-load padding preserves the actual access trace.
The existing source execution equality supplies the very same successor
states, including every dynamic-loop iteration. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.ZeroPadding
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 20000

theorem loadZero_safe {d : TileDType} {sh : TileShape} (x : Op d sh)
    (bounds : RegionBounds) (s : BlockState) :
    (loadZero x).SafeAt bounds s ↔ x.SafeAt bounds s := by
  cases x <;> try rfl
  case load padding mem mask =>
    cases mem <;> try rfl
    simp only [loadZero]
    conv_lhs => rw [Op.SafeAt.eq_def]
    conv_rhs => rw [Op.SafeAt.eq_def]

theorem operation_safe {d : TileDType} {sh : TileShape} (x : Op d sh)
    (bounds : RegionBounds) (s : BlockState) :
    (operation x).SafeAt bounds s ↔ x.SafeAt bounds s := by
  cases x <;> simp only [operation] <;> try exact loadZero_safe _ bounds s
  case add d x y b =>
    conv_lhs => rw [Op.SafeAt.eq_def]
    conv_rhs => rw [Op.SafeAt.eq_def]
    exact and_congr (loadZero_safe x bounds s) (loadZero_safe y bounds s)

mutual
theorem statement_safe (st : Stmt) (bounds : RegionBounds) (s : BlockState)
    (hok : st.FlattenOk) (hu : s.undef=fun _ _=>0) :
    (statement st).TraceSafe bounds s ↔ st.TraceSafe bounds s := by
  cases hst : st <;> simp only [hst] at hok <;> simp only [statement]
  case assign d sh name x => simpa only [Stmt.TraceSafe] using operation_safe x bounds s
  case forRangeDyn name first last stride body =>
    simp only [Stmt.TraceSafe]
    cases evalOp first s <;> cases evalOp last s <;> cases evalOp stride s <;>
      simp only [range_safe name _ _ _ body bounds s hok.2.2.2 hu]
  case ifThen c body =>
    simp only [Stmt.TraceSafe]
    cases evalOp c s with
    | none => rfl
    | some v =>
      by_cases hv : v.data PUnit.unit
      · simp only [hv,↓reduceIte,statements_safe body bounds s hok.2 hu]
      · simp only [hv,Bool.false_eq_true,↓reduceIte]
  case ifThenElse c yes no =>
    simp only [Stmt.TraceSafe]
    cases evalOp c s with
    | none => rfl
    | some v =>
      by_cases hv : v.data PUnit.unit
      · simp only [hv,↓reduceIte,statements_safe yes bounds s hok.2.1 hu]
      · simp only [hv,Bool.false_eq_true,↓reduceIte,statements_safe no bounds s hok.2.2 hu]
termination_by (sizeOf st,0)
decreasing_by all_goals (simp_wf; subst_vars; simp_wf; omega)

theorem statements_safe (sts : List Stmt) (bounds : RegionBounds) (s : BlockState)
    (hok : StmtList.FlattenOk sts) (hu : s.undef=fun _ _=>0) :
    Stmt.TraceSafeList bounds (statements sts) s ↔ Stmt.TraceSafeList bounds sts s := by
  cases hsts : sts with
  | nil => simp only [statements]
  | cons st rest =>
    simp only [hsts,StmtList.FlattenOk] at hok
    rw [statements,Stmt.TraceSafeList,Stmt.TraceSafeList,statement_safe st bounds s hok.1 hu,
      statement_eq st s hok.1 hu]
    cases he : stepStmt st s with
    | none => rfl
    | some t =>
      dsimp only
      rw [statements_safe rest bounds t hok.2 ((stepStmt_undef st s t hok.1 he).trans hu)]
termination_by (sizeOf sts,0)
decreasing_by all_goals (simp_wf; subst_vars; simp_wf; omega)

theorem range_safe (name : RegName) (first last stride : Nat) (body : List Stmt)
    (bounds : RegionBounds) (s : BlockState)
    (hok : StmtList.FlattenOk body) (hu : s.undef=fun _ _=>0) :
    Stmt.forRangeTraceSafe bounds name first last stride (statements body) s ↔
      Stmt.forRangeTraceSafe bounds name first last stride body s := by
  rw [Stmt.forRangeTraceSafe,Stmt.forRangeTraceSafe]
  by_cases hz : stride=0
  · simp only [hz,↓reduceIte]
  · simp only [hz,↓reduceIte]
    by_cases hl : first<last
    · simp only [hl,↓reduceIte]
      have hs : (s.setReg name .nat [] (Tile.scalar first)).undef=fun _ _=>0 := hu
      rw [statements_safe body bounds _ hok hs,statements_eq body _ hok hs]
      cases he : stepStmts body (s.setReg name .nat [] (Tile.scalar first)) with
      | none => rfl
      | some t =>
        dsimp only
        rw [range_safe name (first+stride) last stride body bounds t hok
          ((stepStmts_undef body _ t hok he).trans hs)]
    · simp only [hl,↓reduceIte]
termination_by (sizeOf body,last-first+1)
decreasing_by all_goals (simp_wf; subst_vars; simp_wf; omega)
end

#axiomsClean statements_safe
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.ZeroPadding
