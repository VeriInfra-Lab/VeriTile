import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOGridSafety
import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOResults

namespace VeriTile.Bench.Repairs.MatmulTriton1.IOCorrect
open VeriTile.Triton FP.Structural Revision001 MatmulIO
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem precision {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (io rest P p).Implements (precisionTarget P p) := by
  apply ProgramIO.Implements.intro
  · intro vals limits bound initial _
    exact IOGridSafety.schedule_safe rest P p vals limits bound
      (by have := ready.blockInner.1; omega) _ initial
  · exact IOResults.complete rest P p canonical ready storage geometry ha hb

theorem real (rest : Algebra ℝ) (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (io rest Core.realPrimitives p).Implements (realTarget p) := by
  apply ProgramIO.Implements.intro
  · intro vals limits bound initial _
    exact IOGridSafety.schedule_safe rest Core.realPrimitives p vals limits bound
      (by have := ready.blockInner.1; omega) _ initial
  · intro xs initial hin
    obtain ⟨final,run,outputs,frame⟩ :=
      IOResults.complete rest Core.realPrimitives p canonical ready storage geometry ha hb xs initial hin
    have targets : precisionTarget Core.realPrimitives p xs=realTarget p xs := by
      funext o j
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      change precisionAccum Core.realPrimitives p xs j.1 j.2.1 (Core.steps p)=_
      rw [← IOResults.precision_target rest Core.realPrimitives p xs initial hin]
      exact (Core.logical_realMatmul p initial _ _
        (by have := ready.blockInner.1; omega)).trans
        (IOResults.real_target rest p xs initial hin j.1 j.2.1)
    rw [targets] at outputs
    exact ⟨final,run,outputs,frame⟩

#axiomsClean precision
#axiomsClean real
end
end VeriTile.Bench.Repairs.MatmulTriton1.IOCorrect
