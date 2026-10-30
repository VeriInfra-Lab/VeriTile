import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Kernels
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Schedules

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 8000000

/-- The complete ordinary quotations elaborate to every statement of the retained
schedule AST, including source aliases, both casts and masked stores. -/
theorem quotation_eq (s : Kernels.Schedule) (g : Graph) (p : Parameters) :
    quotation s g p = Kernels.source s g p := by
  cases s <;> cases g <;> rfl

theorem native_execution [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (r : Whole.Request) (initial : State Word) :
    ∃ final, exec (backend rest cfg D F) (compiled r) initial = some final ∧
      final.mem = (written cfg r.parameters initial
        (accumulated cfg D F .accumulator r.parameters initial)).mem := by
  rw [compiled, quotation_eq]
  exact Schedules.execution_memory rest cfg D F r.schedule .accumulator r.parameters initial

#axiomsClean quotation_eq
#axiomsClean native_execution
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
