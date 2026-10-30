import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Proofs.IOSafety
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Proofs.KernelCompatibility

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.IOComplete
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem real_implements (p : KernelInterface.Parameters)
    (separateA : p.C≠p.A) (separateB : p.C≠p.B) (order : Schedule p.request) :
    (io (Whole.idealAlgebra id) Whole.Contracted.idealDotAcc p order).Implements (realReference p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds backed initial _
    exact IOSafety.whole _ _ p order vals bounds backed initial
  · intro vals initial hin
    obtain ⟨final,ran,values,frame⟩ := mathematical_kernel_raw p initial
      (matrixA vals) (matrixB vals)
      (IOFacts.input_a _ _ p order vals initial hin)
      (IOFacts.input_b _ _ p order vals initial hin) separateA separateB order
    refine ⟨final,?_,?_,?_⟩
    · exact (IOFacts.schedule_run _ _ p order.order initial).trans ran
    · intro port j _
      fin_cases port
      change mapCell id (final.mem p.C (Whole.outputOffset p.request j.1.val j.2.1.val)) = _
      rw [values j.1 j.2.1]
      rfl
    · intro r o outside
      exact frame r o ((IOFacts.outside _ _ p order vals r o).mp outside)

theorem native_implements (M : Algebra Word) (D : FPFused.DotAcc Word)
    (Q : Whole.Contracted.Numerics Word) (law : Whole.Contracted.PrimitiveLaws M D Q)
    (p : KernelInterface.Parameters) (separateA : p.C≠p.A) (separateB : p.C≠p.B)
    (order : Schedule p.request) :
    (io M D p order).Implements (nativeReference Q p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds backed initial _
    exact IOSafety.whole _ _ p order vals bounds backed initial
  · intro vals initial hin
    obtain ⟨final,ran,values,frame⟩ := floating_kernel_raw M D Q law p initial
      (matrixA vals) (matrixB vals)
      (IOFacts.input_a _ _ p order vals initial hin)
      (IOFacts.input_b _ _ p order vals initial hin) separateA separateB order
    refine ⟨final,?_,?_,?_⟩
    · exact (IOFacts.schedule_run _ _ p order.order initial).trans ran
    · intro port j _
      fin_cases port
      change mapCell id (final.mem p.C (Whole.outputOffset p.request j.1.val j.2.1.val)) = _
      rw [values j.1 j.2.1]
      rfl
    · intro r o outside
      exact frame r o ((IOFacts.outside _ _ p order vals r o).mp outside)

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.IOComplete
