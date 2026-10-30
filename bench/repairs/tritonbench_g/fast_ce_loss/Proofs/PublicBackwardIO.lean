import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.IOBinding
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.SourceCells
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.SourceAccessBackward

namespace VeriTile.Bench.Repairs.FastCeLoss.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem backward_implements (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (wholeRow : Bool) :
    (backwardIO M id id p row chunk wholeRow).Implements
      (backwardFormula M p chunk wholeRow) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded initial binding
    let start := (backwardInvocation M p row chunk).initial initial
    have ht := backward_label M p row chunk wholeRow xs initial binding
    change ScheduleSafe bounds [backwardInvocation M p row chunk] initial
    rw [scheduleSafe_singleton]
    change TraceSafe (engine M) bounds (KernelSource.backward p) start
    apply SourceAccess.backward_safe M bounds p start
    · exact bounded.2.1 (0 : Fin 4) PUnit.unit trivial
    · intro lane active
      simpa only [start,backward_row,backward_chunk] using
        backward_tile_bounds M p row chunk wholeRow xs bounds bounded lane
          (by simpa only [start,backward_chunk] using active)
    · intro active
      exact bounded.2.1 (3 : Fin 4) PUnit.unit (by simpa only [start,ht] using active)
    · exact bounded.2.1 (2 : Fin 4) PUnit.unit trivial
  · intro xs initial binding
    let start := (backwardInvocation M p row chunk).initial initial
    have ht := backward_label M p row chunk wholeRow xs initial binding
    have values := backward_tile M p row chunk wholeRow xs initial binding
    have saved : (start.mem p.lse (start.pids 0)).read .real=xs (2 : Fin 4) PUnit.unit :=
      binding (2 : Fin 4) PUnit.unit trivial
    have upstream : label p start ≠ -100 →
        (start.mem p.upstream (start.pids 0*p.upstreamStride)).read .real=xs (3 : Fin 4) PUnit.unit := by
      intro active
      exact binding (3 : Fin 4) PUnit.unit (by simpa only [start,ht] using active)
    obtain ⟨final,ran,outputs,frame⟩ := SourceCells.backward M p start
    refine ⟨final,?_,?_,?_⟩
    · simpa only [backwardIO,runSchedule_singleton,Invocation.run,backwardInvocation] using ran
    · intro o j active
      fin_cases o
      have hvalue := LogicalModel.gradient_binding M p start j.1
        (tile p wholeRow chunk (xs (1 : Fin 4)) j.1) (xs (2 : Fin 4) PUnit.unit)
        (xs (3 : Fin 4) PUnit.unit) active (values j.1 active) saved upstream
      change mapCell id (final.mem p.logits (row*p.logitsStride+(chunk*p.block+j.1.val)))=_
      simpa only [start,KernelExecution.gradientOffset,backward_row,backward_chunk,ht] using
        congrArg (mapCell id) ((outputs j.1 active).trans (congrArg (Cell.mk .real) hvalue))
    · intro r o outside
      exact frame r o (fun lane active => outside.1 (0 : Fin 1) (lane,PUnit.unit) active)

#axiomsClean backward_implements
end
end VeriTile.Bench.Repairs.FastCeLoss.PublicIO
