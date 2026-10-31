import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealIteration

/-! The actual FP8 iteration retains its index vector and advances each input
pointer by one key block. These are source-structural facts, independent of the
numeric recurrence and of floating-point admission. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoopFrame
open VeriTile.Triton RealKernel FP8RealIteration
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

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

/-- Launch metadata consumed by the output-address computation. -/
structure Launch (initial : BlockState) (NH : Nat) (s : BlockState) : Prop where
  pids : s.pids = initial.pids
  batch : s.regs .nat [] "batch_id" = some (Tile.scalar (initial.pids 1 / NH))
  head : s.regs .nat [] "off_hz" = some (Tile.scalar (initial.pids 1 % NH))

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

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoopFrame
