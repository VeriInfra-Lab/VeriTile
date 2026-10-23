import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedGridExecution

/-!
Actual source control-flow execution, parameterized only by a locally proved
per-tile statement implementation. The source keeps pid, num_tiles0, num_ctas,
the j loop and tile_id recomputation. Only memory and padding observations are
compared: source scratch registers need not equal the flattened tile executor.
The final source module discharges this local interface with signed perTile.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceForRange
open VeriTile.Triton SignedView SignedTile SignedGrid

def ObsEq (s t : BlockState) : Prop := s.mem = t.mem ∧ s.undef = t.undef

theorem ObsEq.refl (s : BlockState) : ObsEq s s := ⟨rfl, rfl⟩
theorem ObsEq.trans {s t u : BlockState} (h : ObsEq s t) (h' : ObsEq t u) : ObsEq s u :=
  ⟨h.1.trans h'.1, h.2.trans h'.2⟩

theorem setReg_obs (s : BlockState) (name : RegName) (dtype : TileDType)
    (shape : TileShape) (value : Tile dtype shape) : ObsEq (s.setReg name dtype shape value) s :=
  ⟨rfl, rfl⟩

def TileImplementation (input output : View) (n B : Nat) (body : List Stmt) : Prop :=
  ∀ T s, s.regs .nat [] "tile_id" = some (Tile.scalar T) →
    ∃ final, stepStmts body s = some final ∧
      ObsEq final (finalState input output n (T * B) B s) ∧
      final.regs .nat [] "pid" = s.regs .nat [] "pid" ∧
      final.regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas"

def tileIdStmt : Stmt :=
  .assign .nat [] "tile_id"
    (.add .nat .nil (.ref .nat [] "pid")
      (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas")))

def loopBody (body : List Stmt) : List Stmt := tileIdStmt :: body

def iterationStarts (P C B cur remaining : Nat) : List Nat :=
  (List.range' cur remaining).map (fun j => (P + j * C) * B)

theorem iterationStarts_zero (P C B cur : Nat) : iterationStarts P C B cur 0 = [] := rfl

theorem iterationStarts_succ (P C B cur remaining : Nat) :
    iterationStarts P C B cur (remaining + 1) =
      (P + cur * C) * B :: iterationStarts P C B (cur + 1) remaining := by
  simp only [iterationStarts, List.range'_succ, List.map_cons]

/-- Exact forRangeDyn engine against the concrete FP32 tile sequence. Every
source iteration recomputes its own tile_id after the engine writes j. -/
theorem range_execution (input output : View) (n B : Nat) (body : List Stmt)
    (hbody : TileImplementation input output n B body)
    (hcongr : ∀ start s t, ObsEq s t →
      ObsEq (finalState input output n start B s) (finalState input output n start B t))
    (P C stop : Nat) :
    ∀ remaining cur, cur + remaining = stop → ∀ actual flat,
      actual.regs .nat [] "pid" = some (Tile.scalar P) →
      actual.regs .nat [] "num_ctas" = some (Tile.scalar C) → ObsEq actual flat →
      ∃ actualFinal flatFinal,
        stepForRangeAux "j" cur stop 1 (loopBody body) actual = some actualFinal ∧
        runTiles (fun start s => stepStmts (tileBody input output n start B) s)
          (iterationStarts P C B cur remaining) flat = some flatFinal ∧
        ObsEq actualFinal flatFinal ∧
        actualFinal.regs .nat [] "pid" = some (Tile.scalar P) ∧
        actualFinal.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
  intro remaining
  induction remaining with
  | zero =>
    intro cur hcur actual flat hpid hctas hobs
    exact ⟨actual, flat, stepForRangeAux.step_ge one_ne_zero (by omega),
      rfl, hobs, hpid, hctas⟩
  | succ remaining ih =>
    intro cur hcur actual flat hpid hctas hobs
    let atJ := actual.setReg "j" .nat [] (Tile.scalar cur)
    let atTile := atJ.setReg "tile_id" .nat [] (Tile.scalar (P + cur * C))
    have hpidJ : atJ.regs .nat [] "pid" = some (Tile.scalar P) := by simp [atJ, hpid]
    have hctasJ : atJ.regs .nat [] "num_ctas" = some (Tile.scalar C) := by simp [atJ, hctas]
    have hj : atJ.regs .nat [] "j" = some (Tile.scalar cur) := by simp [atJ]
    have hassign : stepStmt tileIdStmt atJ = some atTile := by
      apply stepStmt_assign_eq_some
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpidJ, hctasJ, hj]
      rfl
    obtain ⟨next, hstep, hnextObs, hnextPid, hnextCtas⟩ :=
      hbody (P + cur * C) atTile (by simp [atTile])
    have hbodyStep : stepStmts (loopBody body) atJ = some next := by
      rw [loopBody, stepStmts.cons_some hassign]
      exact hstep
    have hnextObsFlat : ObsEq next (finalState input output n ((P + cur * C) * B) B flat) :=
      hnextObs.trans (hcongr _ _ _ ((setReg_obs _ _ _ _ _).trans
        ((setReg_obs _ _ _ _ _).trans hobs)))
    have hnextP : next.regs .nat [] "pid" = some (Tile.scalar P) := by
      rw [hnextPid]
      simp [atTile, atJ, hpid]
    have hnextC : next.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
      rw [hnextCtas]
      simp [atTile, atJ, hctas]
    obtain ⟨actualFinal, flatFinal, hactualTail, hflatTail, hfinalObs, hfinalP, hfinalC⟩ :=
      ih (cur + 1) (by omega) next
        (finalState input output n ((P + cur * C) * B) B flat)
        hnextP hnextC hnextObsFlat
    refine ⟨actualFinal, flatFinal, ?_, ?_, hfinalObs, hfinalP, hfinalC⟩
    · rw [stepForRangeAux.step_lt one_ne_zero (by omega), hbodyStep, Option.bind_some]
      exact hactualTail
    · rw [iterationStarts_succ, runTiles, tile_exec, Option.bind_some]
      exact hflatTail

/-- The source's otherwise unused cdiv assignment is retained in the program. -/
def numTilesStmt (n B : Nat) : Stmt :=
  .assign .nat [] "num_tiles0"
    (.div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat n) (.constNat B)) (.constNat 1))
      (.constNat B))

theorem numTiles_step (n B : Nat) (s : BlockState) :
    stepStmt (numTilesStmt n B) s =
      some (s.setReg "num_tiles0" .nat [] (Tile.scalar ((n + B - 1) / B))) := by
  apply stepStmt_assign_eq_some
  simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat]
  rfl

def gridBody (body : List Stmt) (n B tiles : Nat) : List Stmt :=
  [.assign .nat [] "pid" (.programId 0), numTilesStmt n B,
   .assign .nat [] "num_ctas" (.numPrograms 0),
   .forRangeDyn "j" (.constNat 0) (.constNat tiles) (.constNat 1) (loopBody body)]

theorem singleton_step (stmt : Stmt) (s : BlockState) :
    stepStmts [stmt] s = stepStmt stmt s := by
  simp only [stepStmts]
  cases stepStmt stmt s <;> rfl

/-- Complete source grid-stride branch, including all three prologue assigns
and the actual dynamic loop. Only the per-tile source proof is parameterized. -/
theorem grid_branch_execution (input output : View) (n B : Nat) (body : List Stmt)
    (hbody : TileImplementation input output n B body)
    (hcongr : ∀ start s t, ObsEq s t →
      ObsEq (finalState input output n start B s) (finalState input output n start B t))
    (tiles : Nat) (s : BlockState) :
    ∃ actualFinal flatFinal,
      stepStmts (gridBody body n B tiles) s = some actualFinal ∧
      runTiles (fun start s => stepStmts (tileBody input output n start B) s)
        ((List.range tiles).map (fun j => (s.pids 0 + j * s.numPids 0) * B)) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  let withPid := s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))
  let withTiles := withPid.setReg "num_tiles0" .nat [] (Tile.scalar ((n + B - 1) / B))
  let withCTAs := withTiles.setReg "num_ctas" .nat [] (Tile.scalar (s.numPids 0))
  have hp : withCTAs.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) := by
    simp [withCTAs, withTiles, withPid]
  have hc : withCTAs.regs .nat [] "num_ctas" = some (Tile.scalar (s.numPids 0)) := by
    simp [withCTAs]
  obtain ⟨actualFinal, flatFinal, hloop, hflat, hobs, _, _⟩ :=
    range_execution input output n B body hbody hcongr (s.pids 0) (s.numPids 0) tiles
      tiles 0 (by omega) withCTAs s hp hc ⟨rfl, rfl⟩
  refine ⟨actualFinal, flatFinal, ?_, ?_, hobs⟩
  · simp only [gridBody]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
      stepStmts.cons_some (numTiles_step n B _),
      stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 0 _)),
      singleton_step, stepForRangeAux.forRangeDyn_unfold]
    simpa only [evalOp_constNat, Option.bind_some] using hloop
  · simpa only [iterationStarts, ← List.range_eq_range'] using hflat

def oneTileBody (body : List Stmt) (n B : Nat) : List Stmt :=
  .assign .nat [] "pid" (.programId 0) :: numTilesStmt n B ::
    .assign .nat [] "tile_id" (.ref .nat [] "pid") :: body

/-- The source one-tile branch retains its pid/cdiv/tile_id prologue, before
running the same dynamically indexed perTile implementation. -/
theorem one_tile_execution (input output : View) (n B : Nat) (body : List Stmt)
    (hbody : TileImplementation input output n B body)
    (hcongr : ∀ start s t, ObsEq s t →
      ObsEq (finalState input output n start B s) (finalState input output n start B t))
    (s : BlockState) :
    ∃ actualFinal,
      stepStmts (oneTileBody body n B) s = some actualFinal ∧
      ObsEq actualFinal (finalState input output n (s.pids 0 * B) B s) := by
  let withPid := s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))
  let withTiles := withPid.setReg "num_tiles0" .nat [] (Tile.scalar ((n + B - 1) / B))
  let withTile := withTiles.setReg "tile_id" .nat [] (Tile.scalar (s.pids 0))
  have hassign : stepStmt (.assign .nat [] "tile_id" (.ref .nat [] "pid")) withTiles =
      some withTile := by
    apply stepStmt_assign_eq_some
    simp [withTiles, withPid]
  obtain ⟨actualFinal, hexec, hobs, _, _⟩ := hbody (s.pids 0) withTile (by simp [withTile])
  refine ⟨actualFinal, ?_, hobs.trans (hcongr _ _ _ ⟨rfl, rfl⟩)⟩
  simp only [oneTileBody]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
    stepStmts.cons_some (numTiles_step n B _), stepStmts.cons_some hassign]
  exact hexec

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceForRange
