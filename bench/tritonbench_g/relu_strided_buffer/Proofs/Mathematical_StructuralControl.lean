import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralGrid
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SourceForRange

/-!
Opaque execution of the original dynamic source loop and CTA prologues.
The local body interface is a tile memory transformer, proved from the actual
source in StructuralTile. No whole-loop or final-output premise is used.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralControl
open VeriTile.Triton FP.Structural StructuralGrid
noncomputable section

abbrev Memory (α : Type) := RegionName → Nat → Cell α

def ObsEq {α : Type} (s t : State α) : Prop := s.mem = t.mem ∧ s.undef = t.undef

theorem ObsEq.refl {α : Type} (s : State α) : ObsEq s s := ⟨rfl, rfl⟩
theorem ObsEq.trans {α : Type} {s t u : State α} (h : ObsEq s t) (h' : ObsEq t u) :
    ObsEq s u := ⟨h.1.trans h'.1, h.2.trans h'.2⟩

def flatState {α : Type} (transform : Nat → Memory α → Memory α) (T : Nat)
    (s : State α) : State α := { s with mem := transform T s.mem }

def flatTile {α : Type} (transform : Nat → Memory α → Memory α) (T : Nat)
    (s : State α) : Option (State α) := some (flatState transform T s)

theorem flatState_obs {α : Type} (transform : Nat → Memory α → Memory α) (T : Nat)
    (s t : State α) (h : ObsEq s t) :
    ObsEq (flatState transform T s) (flatState transform T t) := by
  exact ⟨congrArg (transform T) h.1, h.2⟩

def TileImplementation {α : Type} [Inhabited α] (M : Algebra α)
    (transform : Nat → Memory α → Memory α) (body : List ComputeStmt) : Prop :=
  ∀ T s, s.regs .nat [] "tile_id" = some (fun _ => T) →
    ∃ final, run M body s = some final ∧ ObsEq final (flatState transform T s) ∧
      final.regs .nat [] "pid" = s.regs .nat [] "pid" ∧
      final.regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas"

def iterationIds (P C cur remaining : Nat) : List Nat :=
  (List.range' cur remaining).map (fun j => P + j * C)

theorem iterationIds_succ (P C cur remaining : Nat) :
    iterationIds P C cur (remaining + 1) =
      (P + cur * C) :: iterationIds P C (cur + 1) remaining := by
  simp only [iterationIds, List.range'_succ, List.map_cons]

def loopBody (body : List ComputeStmt) : List ComputeStmt :=
  .alg SourceForRange.tileIdStmt :: body

theorem range_execution {α : Type} [Inhabited α] (M : Algebra α)
    (transform : Nat → Memory α → Memory α) (body : List ComputeStmt)
    (hbody : TileImplementation M transform body) (P C stop : Nat) :
    ∀ remaining cur, cur + remaining = stop → ∀ actual flat,
      actual.regs .nat [] "pid" = some (fun _ => P) →
      actual.regs .nat [] "num_ctas" = some (fun _ => C) → ObsEq actual flat →
      ∃ actualFinal flatFinal,
        FP.Structural.range M "j" cur stop 1 (loopBody body) actual = some actualFinal ∧
        runTiles (flatTile transform) (iterationIds P C cur remaining) flat = some flatFinal ∧
        ObsEq actualFinal flatFinal ∧
        actualFinal.regs .nat [] "pid" = some (fun _ => P) ∧
        actualFinal.regs .nat [] "num_ctas" = some (fun _ => C) := by
  intro remaining
  induction remaining with
  | zero =>
    intro cur hcur actual flat hpid hctas hobs
    refine ⟨actual, flat, ?_, rfl, hobs, hpid, hctas⟩
    rw [FP.Structural.range]
    simp only [one_ne_zero, ↓reduceIte, show ¬cur < stop by omega]
  | succ remaining ih =>
    intro cur hcur actual flat hpid hctas hobs
    let atJ := actual.setReg "j" .nat [] (fun _ => cur)
    let atTile := atJ.setReg "tile_id" .nat [] (fun _ => P + cur * C)
    have hpidJ : atJ.regs .nat [] "pid" = some (fun _ => P) := by simp [atJ, hpid]
    have hctasJ : atJ.regs .nat [] "num_ctas" = some (fun _ => C) := by simp [atJ, hctas]
    have hj : atJ.regs .nat [] "j" = some (fun _ => cur) := by simp [atJ]
    have hassign : step M (.alg SourceForRange.tileIdStmt) atJ = some atTile := by
      simp only [SourceForRange.tileIdStmt, step, stepAlg, evalOp_unfold,
        hpidJ, hctasJ, hj, numeric, bind, pure, Option.bind, atTile]
      rfl
    obtain ⟨next, hstep, hnextObs, hnextPid, hnextCtas⟩ :=
      hbody (P + cur * C) atTile (by simp [atTile])
    have hbodyStep : run M (loopBody body) atJ = some next := by
      simp only [loopBody, run, hassign, bind, Option.bind, hstep]
    have hnextObsFlat : ObsEq next (flatState transform (P + cur * C) flat) :=
      hnextObs.trans (flatState_obs transform _ _ _ hobs)
    have hnextP : next.regs .nat [] "pid" = some (fun _ => P) := by
      rw [hnextPid]
      simp [atTile, atJ, hpid]
    have hnextC : next.regs .nat [] "num_ctas" = some (fun _ => C) := by
      rw [hnextCtas]
      simp [atTile, atJ, hctas]
    obtain ⟨actualFinal, flatFinal, hactualTail, hflatTail, hfinalObs, hfinalP, hfinalC⟩ :=
      ih (cur + 1) (by omega) next (flatState transform (P + cur * C) flat)
        hnextP hnextC hnextObsFlat
    refine ⟨actualFinal, flatFinal, ?_, ?_, hfinalObs, hfinalP, hfinalC⟩
    · rw [FP.Structural.range]
      simp only [one_ne_zero, ↓reduceIte, show cur < stop by omega]
      change (run M (loopBody body) atJ).bind
        (fun t => FP.Structural.range M "j" (cur + 1) stop 1 (loopBody body) t) = _
      rw [hbodyStep]
      exact hactualTail
    · simpa only [iterationIds_succ, runTiles, flatTile, Option.bind_some] using hflatTail

def gridBody (body : List ComputeStmt) (n B tiles : Nat) : List ComputeStmt :=
  [.alg (.assign .nat [] "pid" (.programId 0)), .alg (SourceForRange.numTilesStmt n B),
   .alg (.assign .nat [] "num_ctas" (.numPrograms 0)),
   .forRangeDyn "j" (.constNat 0) (.constNat tiles) (.constNat 1) (loopBody body)]

def oneTileBody (body : List ComputeStmt) (n B : Nat) : List ComputeStmt :=
  .alg (.assign .nat [] "pid" (.programId 0)) :: .alg (SourceForRange.numTilesStmt n B) ::
    .alg (.assign .nat [] "tile_id" (.ref .nat [] "pid")) :: body

theorem numTiles_step {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α) :
    step M (.alg (SourceForRange.numTilesStmt n B)) s =
      some (s.setReg "num_tiles0" .nat [] (fun _ => (n + B - 1) / B)) := by
  simp [SourceForRange.numTilesStmt, step, stepAlg, evalOp_unfold, numeric, bind, pure, Option.bind]
  rfl

theorem grid_branch_execution {α : Type} [Inhabited α] (M : Algebra α)
    (transform : Nat → Memory α → Memory α) (body : List ComputeStmt)
    (hbody : TileImplementation M transform body) (n B tiles : Nat) (s : State α) :
    ∃ actualFinal flatFinal,
      run M (gridBody body n B tiles) s = some actualFinal ∧
      runTiles (flatTile transform)
        ((List.range tiles).map (fun j => s.pids 0 + j * s.numPids 0)) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  let withPid := s.setReg "pid" .nat [] (fun _ => s.pids 0)
  let withTiles := withPid.setReg "num_tiles0" .nat [] (fun _ => (n + B - 1) / B)
  let withCTAs := withTiles.setReg "num_ctas" .nat [] (fun _ => s.numPids 0)
  have hp : withCTAs.regs .nat [] "pid" = some (fun _ => s.pids 0) := by
    simp [withCTAs, withTiles, withPid]
  have hc : withCTAs.regs .nat [] "num_ctas" = some (fun _ => s.numPids 0) := by
    simp [withCTAs]
  obtain ⟨actualFinal, flatFinal, hloop, hflat, hobs, _, _⟩ :=
    range_execution M transform body hbody (s.pids 0) (s.numPids 0) tiles
      tiles 0 (by omega) withCTAs s hp hc ⟨rfl, rfl⟩
  refine ⟨actualFinal, flatFinal, ?_, ?_, hobs⟩
  · simp only [gridBody, run, step, stepAlg, evalOp_unfold, bind, pure, Option.bind,
      SourceForRange.numTilesStmt, numeric]
    change (FP.Structural.range M "j" 0 tiles 1 (loopBody body) withCTAs).bind
      (fun t => some t) = some actualFinal
    rw [hloop]
    rfl
  · simpa only [iterationIds, ← List.range_eq_range'] using hflat

theorem one_tile_execution {α : Type} [Inhabited α] (M : Algebra α)
    (transform : Nat → Memory α → Memory α) (body : List ComputeStmt)
    (hbody : TileImplementation M transform body) (n B : Nat) (s : State α) :
    ∃ actualFinal,
      run M (oneTileBody body n B) s = some actualFinal ∧
      ObsEq actualFinal (flatState transform (s.pids 0) s) := by
  let withPid := s.setReg "pid" .nat [] (fun _ => s.pids 0)
  let withTiles := withPid.setReg "num_tiles0" .nat [] (fun _ => (n + B - 1) / B)
  let withTile := withTiles.setReg "tile_id" .nat [] (fun _ => s.pids 0)
  obtain ⟨actualFinal, hexec, hobs, _, _⟩ := hbody (s.pids 0) withTile (by simp [withTile])
  refine ⟨actualFinal, ?_, hobs⟩
  simpa only [oneTileBody, run, step, stepAlg, evalOp_unfold, bind, pure, Option.bind,
    SourceForRange.numTilesStmt, numeric, State.setReg_ne_name,
    State.setReg_same, withTile, withTiles, withPid] using hexec

def atProgram {α : Type} (C pid : Nat) (s : State α) : State α :=
  { s with
    pids := fun d => if d = 0 then pid else 0
    numPids := fun d => if d = 0 then C else 1 }

def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt)
    (C : Nat) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid :: rest, s => do
    let next ← run M body (atProgram C pid s)
    runPrograms M body C rest next

theorem tiles_obs {α : Type} (transform : Nat → Memory α → Memory α) (tiles : List Nat)
    (s t : State α) (h : ObsEq s t) :
    ∃ sf tf, runTiles (flatTile transform) tiles s = some sf ∧
      runTiles (flatTile transform) tiles t = some tf ∧ ObsEq sf tf := by
  induction tiles generalizing s t with
  | nil => exact ⟨s, t, rfl, rfl, h⟩
  | cons tile rest ih =>
    obtain ⟨sf, tf, hs, ht, ho⟩ := ih (flatState transform tile s)
      (flatState transform tile t) (flatState_obs transform tile s t h)
    exact ⟨sf, tf, by simpa only [runTiles, flatTile, Option.bind_some] using hs,
      by simpa only [runTiles, flatTile, Option.bind_some] using ht, ho⟩

theorem programs_execution {α : Type} [Inhabited α] (M : Algebra α)
    (transform : Nat → Memory α → Memory α) (C : Nat) (body : List ComputeStmt)
    (ids : Nat → List Nat)
    (hbody : ∀ pid s, ∃ actualFinal flatFinal,
      run M body (atProgram C pid s) = some actualFinal ∧
      runTiles (flatTile transform) (ids pid) (atProgram C pid s) = some flatFinal ∧
      ObsEq actualFinal flatFinal)
    (pids : List Nat) (actual flat : State α) (hobs : ObsEq actual flat) :
    ∃ actualFinal flatFinal,
      runPrograms M body C pids actual = some actualFinal ∧
      runTiles (flatTile transform) (pids.flatMap ids) flat = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  induction pids generalizing actual flat with
  | nil => exact ⟨actual, flat, rfl, rfl, hobs⟩
  | cons pid rest ih =>
    obtain ⟨actualNext, localFlat, hactual, hlocal, hlocalObs⟩ := hbody pid actual
    obtain ⟨localFlat', flatNext, hlocal', hflat, hflatObs⟩ :=
      tiles_obs transform (ids pid) (atProgram C pid actual) flat hobs
    have hsame : localFlat = localFlat' := Option.some.inj (hlocal.symm.trans hlocal')
    subst localFlat'
    obtain ⟨actualFinal, flatFinal, hactualTail, hflatTail, hfinal⟩ :=
      ih actualNext flatNext (hlocalObs.trans hflatObs)
    refine ⟨actualFinal, flatFinal, ?_, ?_, hfinal⟩
    · simpa only [runPrograms, hactual, Option.bind_some] using hactualTail
    · simpa only [List.flatMap_cons, runTiles_append, hflat, Option.bind_some] using hflatTail

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralControl
