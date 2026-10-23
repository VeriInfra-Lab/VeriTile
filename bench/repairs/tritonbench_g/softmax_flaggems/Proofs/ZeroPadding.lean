import bench.repairs.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.ZeroPadding
open VeriTile Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- A proof transport invariant for the old, explicitly parameterized padding
semantics. The repaired source supplies the zero itself at every masked load. -/
def Zero (s : BlockState) : Prop := s.undef = fun _ _ => 0

def Equivalent (old repaired : List Stmt) : Prop :=
  ∀ s, Zero s → stepStmts old s = stepStmts repaired s

theorem refl (body : List Stmt) : Equivalent body body := fun _ _ => rfl

theorem cons (a b : Stmt) (old repaired : List Stmt)
    (ha : ∀ s, Zero s → stepStmt a s = stepStmt b s)
    (hok : a.FlattenOk) (ht : Equivalent old repaired) :
    Equivalent (a :: old) (b :: repaired) := by
  intro s hs
  simp only [stepStmts]
  rw [← ha s hs]
  cases h : stepStmt a s with
  | none => rfl
  | some t =>
    exact ht t ((stepStmt_undef a s t hok h).trans hs)

theorem range (idx : RegName) (cur stop stride : Nat)
    (old repaired : List Stmt) (hok : StmtList.FlattenOk old)
    (hb : Equivalent old repaired) (s : BlockState) (hs : Zero s) :
    stepForRangeAux idx cur stop stride old s =
      stepForRangeAux idx cur stop stride repaired s := by
  by_cases hz : stride = 0
  · subst stride
    simp
  · by_cases hc : cur < stop
    · rw [stepForRangeAux.step_lt hz hc, stepForRangeAux.step_lt hz hc,
        ← hb (s.setReg idx .nat [] (Tile.scalar cur)) hs]
      cases h : stepStmts old (s.setReg idx .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some t =>
        exact range idx (cur + stride) stop stride old repaired hok hb t
          ((stepStmts_undef old _ t hok h).trans hs)
    · rw [stepForRangeAux.step_ge hz (by omega), stepForRangeAux.step_ge hz (by omega)]
termination_by stop - cur
decreasing_by omega

theorem dynamic_range (idx : RegName) (start stop stride : Op .nat [])
    (old repaired : List Stmt) (hok : StmtList.FlattenOk old)
    (hb : Equivalent old repaired) (s : BlockState) (hs : Zero s) :
    stepStmt (.forRangeDyn idx start stop stride old) s =
      stepStmt (.forRangeDyn idx start stop stride repaired) s := by
  simp only [stepForRangeAux.forRangeDyn_unfold]
  congr 1
  funext a
  congr 1
  funext b
  congr 1
  funext c
  exact range _ _ _ _ _ _ hok hb s hs

theorem static_range (idx : RegName) (start stop stride : Nat)
    (old repaired : List Stmt) (hok : StmtList.FlattenOk old)
    (hb : Equivalent old repaired) (s : BlockState) (hs : Zero s) :
    stepStmt (.forRange idx start stop stride old) s =
      stepStmt (.forRange idx start stop stride repaired) s := by
  simp only [stepForRangeAux.forRange_unfold]
  exact range _ _ _ _ _ _ hok hb s hs

theorem branch (condition : Op .bool []) (a b c d : List Stmt)
    (hab : Equivalent a b) (hcd : Equivalent c d)
    (s : BlockState) (hs : Zero s) :
    stepStmt (.ifThenElse condition a c) s =
      stepStmt (.ifThenElse condition b d) s := by
  simp only [stepStmt]
  cases h : evalOp condition s with
  | none => rfl
  | some value =>
    simp only [bind, Option.bind]
    split <;> first | exact hab s hs | exact hcd s hs

theorem assign (dtype : TileDType) (shape : TileShape) (name : RegName)
    (a b : Op dtype shape) (h : ∀ s, Zero s → evalOp a s = evalOp b s)
    (s : BlockState) (hs : Zero s) :
    stepStmt (.assign dtype shape name a) s = stepStmt (.assign dtype shape name b) s := by
  simp only [stepStmt, h s hs]

theorem load_ptr (shape : TileShape) (p : Op .ptr shape) (m : Op .bool shape)
    (padding : PaddingOption) (s : BlockState) (hs : Zero s) :
    evalOp (.load .real (.ptr p) (.mask m) padding) s =
      evalOp (.load .real (.ptr p) (.maskOther m ((Op.const 0).broadcast shape)) padding) s := by
  simp only [evalOp]
  cases hp : evalOp p s <;> cases hm : evalOp m s <;>
    simp only [bind, Option.bind]
  congr 2
  funext i
  split <;> simp_all [Zero]

theorem load_region (shape : TileShape) (r : Region .real) (off : Op .nat shape)
    (m : Op .bool shape) (padding : PaddingOption) (s : BlockState) (hs : Zero s) :
    evalOp (.load .real (.region r off) (.mask m) padding) s =
      evalOp (.load .real (.region r off) (.maskOther m ((Op.const 0).broadcast shape)) padding) s := by
  simp only [evalOp]
  cases hp : evalOp off s <;> cases hm : evalOp m s <;>
    simp only [bind, Option.bind]
  congr 2
  funext i
  split <;> simp_all [Zero]

end VeriTile.Bench.Repairs.SoftmaxFlaggems.ZeroPadding
