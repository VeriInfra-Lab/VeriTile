import bench.tritonbench_g.relu_strided_buffer.Proofs.SourceBranch
import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedGridExecution

/-!
Composition of complete source CTA bodies. A CTA starts with the source grid
metadata. Private scratch registers are not compared to the flattened executor;
only memory and unspecified-padding observations cross that boundary. The local
body interface is discharged by the source branch/loop theorem in the host module.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceLaunch
open VeriTile.Triton SignedView SignedTile SignedGrid SignedGridExecution SourceBranch

def atProgram (C pid : Nat) (s : BlockState) : BlockState :=
  { s with
    pids := fun d => if d = 0 then pid else 0
    numPids := fun d => if d = 0 then C else 1 }

theorem atProgram_obs (C pid : Nat) (s : BlockState) : ObsEq (atProgram C pid s) s :=
  ⟨rfl, rfl⟩

noncomputable def runPrograms (body : List Stmt) (C : Nat) : List Nat → BlockState → Option BlockState
  | [], s => some s
  | pid :: rest, s => do
    let next ← stepStmts body (atProgram C pid s)
    runPrograms body C rest next

theorem tiles_obs (input output : View) (n B : Nat) (starts : List Nat)
    (s t : BlockState) (h : ObsEq s t) :
    ∃ sf tf,
      runTiles (fun start state => stepStmts (tileBody input output n start B) state)
        starts s = some sf ∧
      runTiles (fun start state => stepStmts (tileBody input output n start B) state)
        starts t = some tf ∧ ObsEq sf tf := by
  induction starts generalizing s t with
  | nil => exact ⟨s, t, rfl, rfl, h⟩
  | cons start rest ih =>
    obtain ⟨sf, tf, hs, ht, ho⟩ := ih
      (finalState input output n start B s) (finalState input output n start B t)
      (tile_obs_congr input output n start B h)
    exact ⟨sf, tf, by simpa only [runTiles, tile_exec, Option.bind_some] using hs,
      by simpa only [runTiles, tile_exec, Option.bind_some] using ht, ho⟩

/-- A local program is run, rather than replaced by its promised output.
Its per-CTA proof is enough to compose any list of actual CTA executions. -/
theorem programs_execution (input output : View) (n B C : Nat) (body : List Stmt)
    (starts : Nat → List Nat)
    (hbody : ∀ pid s, ∃ actualFinal flatFinal,
      stepStmts body (atProgram C pid s) = some actualFinal ∧
      runTiles (fun start state => stepStmts (tileBody input output n start B) state)
        (starts pid) (atProgram C pid s) = some flatFinal ∧
      ObsEq actualFinal flatFinal)
    (pids : List Nat) (actual flat : BlockState) (hobs : ObsEq actual flat) :
    ∃ actualFinal flatFinal,
      runPrograms body C pids actual = some actualFinal ∧
      runTiles (fun start state => stepStmts (tileBody input output n start B) state)
        (pids.flatMap starts) flat = some flatFinal ∧ ObsEq actualFinal flatFinal := by
  induction pids generalizing actual flat with
  | nil => exact ⟨actual, flat, rfl, rfl, hobs⟩
  | cons pid rest ih =>
    obtain ⟨actualNext, localFlat, hactual, hlocal, hlocalObs⟩ := hbody pid actual
    obtain ⟨localFlat', flatNext, hlocal', hflat, hflatObs⟩ :=
      tiles_obs input output n B (starts pid) (atProgram C pid actual) flat
        ((atProgram_obs C pid actual).trans hobs)
    have hsame : localFlat = localFlat' := Option.some.inj (hlocal.symm.trans hlocal')
    subst localFlat'
    obtain ⟨actualFinal, flatFinal, hactualTail, hflatTail, hfinal⟩ :=
      ih actualNext flatNext (hlocalObs.trans hflatObs)
    refine ⟨actualFinal, flatFinal, ?_, ?_, hfinal⟩
    · simpa only [runPrograms, hactual, Option.bind_some] using hactualTail
    · simpa only [List.flatMap_cons, runTiles_append, hflat, Option.bind_some] using hflatTail

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceLaunch
