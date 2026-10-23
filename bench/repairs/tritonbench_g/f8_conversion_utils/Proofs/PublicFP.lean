import bench.repairs.tritonbench_g.f8_conversion_utils.IO
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.KernelContract
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.SourceSafety
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.F8ConversionUtils.PublicFP
open VeriTile.Triton FP.Structural Source.BitAlgebra
open Checked
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

theorem implements (policy : Numerics.Policy) (p : Source.Profile) (Y X : RegionName)
    (N B pid : Nat) :
    (io policy p Y X N B pid).Implements (target policy p Y X N B pid) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bound s input
    rw [show (io policy p Y X N B pid).schedule vals = [_] from rfl,
      scheduleSafe_singleton]
    apply Source.trace_safe
    · intro j active
      exact bound.2.1 0 (j,()) active
    · intro j active
      exact bound.2.2.1 0 (j,()) active
  · intro vals s input
    let initial : State Scalar := { s with pids := fun ax => if ax=0 then pid else 0 }
    obtain ⟨final,run,memory⟩ := Source.executes
      ((KernelContract.model policy p).withDefaultPrecision .fp32) p Y X N B initial
    refine ⟨final,?_,?_,?_⟩
    · simpa only [io,runSchedule_singleton,Invocation.run] using run
    · intro o
      fin_cases o
      intro j active
      have read := input 0 j active
      change (s.mem X (pid*B+j.1.val)).read p.input.toTileDType =
        mapValue id p.input.toTileDType (vals 0 j) at read
      have value := Source.memory_readback
        ((KernelContract.model policy p).withDefaultPrecision .fp32) p Y X N B initial j.1 active
      change mapCell id (final.mem Y (initial.pids 0*B+j.1.val)) =
        Cell.mk p.output.toTileDType
          (ofFloat p.output (formula policy p (toFloat p.input (vals 0 j))))
      rw [memory,value]
      cases p <;> simp_all [io,target,formula,KernelContract.model,Source.cast,algebra,
        Source.Profile.input,Source.Profile.output,Algebra.withDefaultPrecision,
        initial,Invocation.initial,bindRegisters,ofFloat,toFloat,mapCell,mapValue]
    · intro r off outside
      rw [memory]
      apply Source.memory_frame
      by_cases eq : r=Y
      · right
        intro j active he
        have := outside.1 0 (j,()) active
        exact this.elim (fun hn => hn eq) (fun hn => hn he)
      · exact Or.inl eq

#axiomsClean implements
end
end VeriTile.Bench.Repairs.F8ConversionUtils.PublicFP
