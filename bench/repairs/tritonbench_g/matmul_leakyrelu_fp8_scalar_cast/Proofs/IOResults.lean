import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.IOImplements
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.FormulaCongruence

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.IOResults
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution ReadProjection
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem real (M : Algebra ℝ) (law : ScalarLaws M realOps)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b) :
    (IO.matmulIO M realOps g b).Implements (fun vals => IO.result g
      (Target.mathematical g (IO.matrixA g vals) (IO.matrixB g vals))) := by
  apply IOImplements.implements M realOps law g b L
  intro vals initial hin i hi j hj
  rw [source_run_real]
  exact (Real.output g.problem (views (b.views g)) (ReadProjection.layout g.problem (b.views g) L)
    (decode (b.views g) initial) i j hi hj).trans
    (FormulaCongruence.mathematical M g b vals initial hin i j hi hj)

theorem native_ops (M : Algebra α) (ops : Operations α) (law : ScalarLaws M ops)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b)
    (positiveK : g.K≠0) :
    (IO.matmulIO M ops g b).Implements (fun vals => IO.result g
      (IO.native ops g (IO.matrixA g vals) (IO.matrixB g vals))) := by
  apply IOImplements.implements M ops law g b L
  intro vals initial hin i hi j hj
  have grid := grid_reference ops ((FormulaCongruence.virtualBuffers b).arguments g)
    ((FormulaCongruence.virtualBuffers b).views g).C
    (by simpa only [KernelBuffers.arguments_problem] using ReadProjection.layout g.problem (b.views g) L)
    (decode (b.views g) initial) positiveK i j hi hj
  have target := FormulaCongruence.native M ops g b vals initial hin i j hi hj
  simpa only [KernelBuffers.arguments_problem] using grid.trans target

theorem native (M : Algebra Word) (ops : PrimitiveOps) (instruction : InstructionProfile)
    (ports : LocalPorts ops instruction) (law : ScalarLaws M ops.operations)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b)
    (positiveK : g.K≠0) :
    (IO.matmulIO M ops.operations g b).Implements (fun vals => IO.result g
      (IO.native (fpReference instruction) g (IO.matrixA g vals) (IO.matrixB g vals))) := by
  simpa only [operations_eq ports] using native_ops M ops.operations law g b L positiveK

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.IOResults
