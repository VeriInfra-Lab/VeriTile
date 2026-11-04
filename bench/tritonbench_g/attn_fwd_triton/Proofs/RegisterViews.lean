import bench.tritonbench_g.attn_fwd_triton.Proofs.ControlView
import Mathlib.Tactic

/-! Register views for hygienically expanded helpers. These lemmas transport
execution through a bijective register renaming; they impose no arithmetic
identities and do not replace any instruction in the source kernel. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.RegisterViews
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

def pull (names : RegName ≃ RegName) (s : BlockState) : BlockState :=
  {s with regs:=fun d sh name=>s.regs d sh (names name)}

@[simp] theorem pull_setReg (names : RegName ≃ RegName) (s : BlockState)
    (name : RegName) (d : TileDType) (sh : TileShape) (value : Tile d sh) :
    pull names (s.setReg (names name) d sh value) =
      (pull names s).setReg name d sh value := by
  ext
  all_goals simp [pull,BlockState.setReg,names.injective.eq_iff]

def Corresponds (names : RegName ≃ RegName) (view actual : List Stmt) : Prop :=
  ∀ s, (stepStmts actual s).map (pull names)=stepStmts view (pull names s)

theorem nil (names : RegName ≃ RegName) : Corresponds names [] [] := by
  intro s
  simp [stepStmts]

theorem assign (names : RegName ≃ RegName) (name : RegName)
    {d : TileDType} {sh : TileShape} (view actual : Op d sh)
    (expression : ∀ s, evalOp actual s=evalOp view (pull names s))
    (before after : List Stmt) (rest : Corresponds names before after) :
    Corresponds names (.assign d sh name view::before)
      (.assign d sh (names name) actual::after) := by
  intro s
  simp only [stepStmts,stepStmt]
  rw [←expression]
  cases h : evalOp actual s with
  | none=>simp [h,bind,Option.bind]
  | some value=>simpa only [h,bind,Option.bind,pull_setReg] using
      (rest (s.setReg (names name) d sh value))

theorem append (names : RegName ≃ RegName) (a b x y : List Stmt)
    (left : Corresponds names a x) (right : Corresponds names b y) :
    Corresponds names (a++b) (x++y) := by
  intro s
  rw [ProgramRewrite.append,ProgramRewrite.append,←left s]
  cases h : stepStmts x s <;> simp only [h,Option.map_none,Option.map_some,
    bind,Option.bind,Option.map_bind]
  exact right _

/-- Signed loop bounds and stride are unchanged by the register view. Both
successful and failed executions are retained. -/
theorem signed_range (names : RegName ≃ RegName) (index : RegName)
    (view actual : List Stmt) (body : Corresponds names view actual)
    (cur stop : Int) (stride : Nat) (s : BlockState) :
    (stepForRangeIntAux (names index) cur stop stride actual s).map (pull names)=
      stepForRangeIntAux index cur stop stride view (pull names s) := by
  generalize hm : (stop-cur).toNat=remaining
  induction remaining using Nat.strong_induction_on generalizing cur s with
  | h remaining ih=>
    by_cases hz : stride=0
    · subst stride
      simp
    by_cases hc : cur<stop
    · simp only [stepForRangeIntAux.step_lt hz hc]
      have matched := body (s.setReg (names index) .int [] (Tile.scalar cur))
      rw [pull_setReg] at matched
      cases ran : stepStmts actual (s.setReg (names index) .int [] (Tile.scalar cur)) with
      | none=>
        simp only [ran,Option.map_none] at matched
        rw [←matched]
        rfl
      | some next=>
        simp only [ran,Option.map_some] at matched
        rw [←matched]
        simp only [Option.bind_some]
        exact ih (stop-(cur+stride)).toNat (by omega) (cur+stride) next rfl
    · simp [stepForRangeIntAux.step_ge hz (by omega : stop≤cur)]

#axiomsClean pull_setReg
#axiomsClean assign
#axiomsClean append
#axiomsClean signed_range
end
end VeriTile.Bench.Repairs.AttnFwdTriton.RegisterViews
