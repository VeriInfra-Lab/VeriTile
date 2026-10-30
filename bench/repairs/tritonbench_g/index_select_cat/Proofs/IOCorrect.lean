import bench.repairs.tritonbench_g.index_select_cat.Proofs.LogicalIO
import bench.repairs.tritonbench_g.index_select_cat.IO

namespace VeriTile.Bench.Repairs.IndexSelectCat.IOCorrect
open VeriTile.Triton FP.Structural Correct
noncomputable section

theorem correct (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat)
    (unique : LogicalIO.Unique p rowPid colPid) :
    Spec.Real ((gatherIO rest p rowPid colPid).Implements (gathered rest p rowPid colPid)) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial binding
    change ScheduleSafe bounds [_] initial
    rw [scheduleSafe_singleton]
    apply LogicalIO.safety rest p rowPid colPid
      (fun i => vals (0 : Fin 2) (i, ())) initial bounds
    · intro i active
      exact binding (0 : Fin 2) (i, ()) active
    · intro i active
      exact bounded.2.1 (0 : Fin 2) (i, ()) active
    · intro i active
      exact bounded.2.1 (1 : Fin 2) i active
    · intro i active
      exact bounded.2.2.1 (0 : Fin 1) i active
  · intro vals initial binding
    obtain ⟨final, run, values, frame⟩ := LogicalIO.complete rest p rowPid colPid
      (fun i => vals (0 : Fin 2) (i, ())) (vals (1 : Fin 2)) initial
      (fun i active => binding (0 : Fin 2) (i, ()) active)
      (fun i active => binding (1 : Fin 2) i active) unique
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [gatherIO, runSchedule_singleton, Invocation.run, Invocation.initial,
        bindRegisters, List.foldl_nil, Option.getD_none, LogicalIO.initial] using run
    · intro port i active
      have eq : port = (0 : Fin 1) := Fin.eq_zero port
      subst port
      change mapCell id (final.mem p.output (LogicalIO.write p rowPid colPid i)) = _
      rw [values i active]
      rfl
    · intro r o outside
      exact frame r o (outside.1 (0 : Fin 1))

#axiomsClean correct
end
end VeriTile.Bench.Repairs.IndexSelectCat.IOCorrect
