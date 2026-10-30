import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Semantics.BlockPadding

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.ZeroPadding
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Only block loads and the merge's enclosing addition change. Ordinary
masked pointer loads, all arithmetic order, addresses and stores are retained. -/
def loadZero : {dtype : TileDType} → {shape : TileShape} → Op dtype shape → Op dtype shape
  | _, _, .load d (.blockPtr p axes) mask _ => .load d (.blockPtr p axes) mask .zero
  | _, _, x => x

def operation : {dtype : TileDType} → {shape : TileShape} → Op dtype shape → Op dtype shape
  | _, _, .add d b x y => .add d b (loadZero x) (loadZero y)
  | _, _, x => loadZero x

theorem loadZero_eval {dtype : TileDType} {shape : TileShape} (x : Op dtype shape)
    (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    evalOp (loadZero x) s = evalOp x s := by
  cases x <;> try rfl
  case load padding mem mask =>
    cases mem <;> try rfl
    simp only [loadZero]
    exact (evalOp_load_padding_of_undef_zero _ _ _ _ s hu).symm

theorem operation_eval {dtype : TileDType} {shape : TileShape} (x : Op dtype shape)
    (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    evalOp (operation x) s = evalOp x s := by
  cases x <;> simp only [operation] <;> try exact loadZero_eval _ s hu
  case add d x y b =>
    simp only [operation, evalOp, loadZero_eval x s hu, loadZero_eval y s hu]

mutual
def statement : Stmt → Stmt
  | .assign d sh name x => .assign d sh name (operation x)
  | .forRangeDyn name first last stride body =>
      .forRangeDyn name first last stride (statements body)
  | .ifThen c body => .ifThen c (statements body)
  | .ifThenElse c yes no => .ifThenElse c (statements yes) (statements no)
  | x => x
termination_by x => sizeOf x
def statements : List Stmt → List Stmt
  | [] => []
  | x :: xs => statement x :: statements xs
termination_by xs => sizeOf xs
end

mutual
theorem statement_eq (st : Stmt) (s : BlockState)
    (hok : st.FlattenOk) (hu : s.undef = fun _ _ => 0) :
    stepStmt (statement st) s = stepStmt st s := by
  cases hst : st <;> simp only [hst] at hok <;> simp only [statement]
  case assign d sh name x =>
    simp only [statement, stepStmt, operation_eval x s hu]
  case forRangeDyn name first last stride body =>
    simp only [statement, stepStmt]
    congr 1
    funext a
    congr 1
    funext b
    congr 1
    funext c
    exact range_eq name _ _ _ body s hok.2.2.2 hu
  case ifThen c body =>
    simp only [statement, stepStmt]
    congr 1
    funext value
    split
    · exact statements_eq body s hok.2 hu
    · rfl
  case ifThenElse c yes no =>
    simp only [statement, stepStmt]
    congr 1
    funext value
    split
    · exact statements_eq yes s hok.2.1 hu
    · exact statements_eq no s hok.2.2 hu
termination_by (sizeOf st, 0)
decreasing_by all_goals (simp_wf; simp_all; omega)

theorem statements_eq (sts : List Stmt) (s : BlockState)
    (hok : StmtList.FlattenOk sts) (hu : s.undef = fun _ _ => 0) :
    stepStmts (statements sts) s = stepStmts sts s := by
  cases hsts : sts with
  | nil => simp only [statements]
  | cons st rest =>
    simp only [hsts, StmtList.FlattenOk] at hok
    rw [statements, stepStmts, statement_eq st s hok.1 hu]
    rw [stepStmts]
    cases he : stepStmt st s with
    | none => rfl
    | some t =>
      exact statements_eq rest t hok.2 ((stepStmt_undef st s t hok.1 he).trans hu)
termination_by (sizeOf sts, 0)
decreasing_by all_goals (simp_wf; simp_all; omega)

theorem range_eq (name : RegName) (first last stride : Nat) (body : List Stmt)
    (s : BlockState) (hok : StmtList.FlattenOk body) (hu : s.undef = fun _ _ => 0) :
    stepForRangeAux name first last stride (statements body) s =
      stepForRangeAux name first last stride body s := by
  rw [stepForRangeAux, stepForRangeAux]
  by_cases hz : stride = 0
  · simp only [hz, if_pos]
  · simp only [hz, if_false]
    by_cases hl : first < last
    · simp only [hl, if_pos]
      have hs : (s.setReg name .nat [] (Tile.scalar first)).undef = fun _ _ => 0 := hu
      rw [statements_eq body _ hok hs]
      cases he : stepStmts body (s.setReg name .nat [] (Tile.scalar first)) with
      | none => rfl
      | some t =>
        exact range_eq name (first + stride) last stride body t hok
          ((stepStmts_undef body _ t hok he).trans hs)
    · simp only [hl, if_false]
termination_by (sizeOf body, last - first + 1)
decreasing_by all_goals (simp_wf; simp_all; omega)
end

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.ZeroPadding
