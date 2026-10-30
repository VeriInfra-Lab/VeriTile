import bench.tritonbench_g.matmul_leakyrelu.Proofs.OriginalIOSafety
import bench.tritonbench_g.matmul_leakyrelu.Proofs.Mathematical_RealModel

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.OriginalReal
open VeriTile.Triton FP.Structural Interface Original
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

theorem scheduled_grid {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (Q : Whole.Numerics α)
    (law : Whole.PrimitiveLaws M D Q) (r : Whole.Request)
    (order : Schedule r) (initial : State α) (ha : r.C ≠ r.A) (hb : r.C ≠ r.B) :
    ∃ final, Whole.runPrograms M D r order.order initial = some final ∧
      Whole.Result Q r initial final := by
  obtain ⟨final,ran,_,_,frame,_,values⟩ := Whole.programs_correct M D Q law r
    initial initial order.order ha hb (fun _ => rfl) (fun _ => rfl)
  refine ⟨final,ran,?_,frame⟩
  intro i hi j hj
  obtain ⟨pid,hp,lane,hr,hc⟩ := Whole.cell_covered r initial i j hi hj
  exact values i hi j hj ⟨pid,order.complete.mem_iff.mpr (List.mem_range.mpr hp),lane,hr,hc⟩

theorem real_implements (p : KernelInterface.Parameters)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) (order : Schedule p.request) :
    (Original.io (Whole.idealAlgebra id) Whole.idealDot p order).Implements (realReference p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds backed initial _
    exact OriginalIOSafety.whole _ _ p order vals bounds backed initial
  · intro vals initial hin
    obtain ⟨final,ran,values,frame⟩ := scheduled_grid (Whole.idealAlgebra id)
      Whole.idealDot (Whole.idealNumerics id) (Whole.ideal_primitives id)
      p.request order initial separateA separateB
    refine ⟨final,(OriginalFacts.schedule_run _ _ p order.order initial).trans ran,?_,?_⟩
    · intro port j _
      fin_cases port
      change mapCell id (final.mem p.C (Whole.outputOffset p.request j.1.val j.2.1.val)) = _
      have value := values j.1.val j.1.isLt j.2.1.val j.2.1.isLt
      rw [Whole.ideal_reference id p.request initial (matrixA vals) (matrixB vals)
        (OriginalFacts.input_a _ _ p order vals initial hin)
        (OriginalFacts.input_b _ _ p order vals initial hin) j.1 j.2.1] at value
      erw [value]
      rfl
    · intro r o outside
      exact frame r o ((OriginalFacts.outside _ _ p order vals r o).mp outside)

#axiomsClean scheduled_grid
#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.OriginalReal
