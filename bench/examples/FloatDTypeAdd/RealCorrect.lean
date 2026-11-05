import bench.examples.FloatDTypeAdd.IO
import bench.examples.FloatDTypeAdd.Kernels
import bench.examples.FloatDTypeAdd.Proofs.Real
import bench.examples.VectorAdd.RealCorrect
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.FloatDTypeAddCorrect
open VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₂

specification float_add_correctness (B : Nat) :
    Spec.Real (floatAddIO B ⊨ fun xs ys i => xs i + ys i) := by
  have halg : (floatAddIO B).kernel.toAlgKernel =
      (VectorAdd.Kernels.addKernel "x" "y" "out" B).toAlgKernel := by
    simp [floatAddIO, ComputeKernel.toAlgKernel, real_projection]
  refine KernelIO₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]
    exact VectorAdd.addKernel_flattenOk "x" "y" "out" B
  · intro bounds s hx hy hout
    rw [halg]
    exact VectorAdd.addKernel_traceSafe "x" "y" "out" B bounds s hx hy hout
  · intro s xs ys hx hy
    rw [halg]
    rcases Nat.eq_zero_or_pos B with rfl | hB
    · simp [floatAddIO, VectorAdd.Kernels.addKernel, exec, stepStmts, stepStmt,
        evalOp.eq_def, Tile.bop, NumericDType.add, NumericDType.mul,
        TileShape.allIndices, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeDType.eraseDType]
    · exact VectorAdd.addKernel_region_run B hB s xs ys hx hy

/-- Both loaded inputs are added, including when the tile is empty. -/
specification float_add_optimized_correctness (B : Nat) :
    Spec.Real (optimizedIO B ⊨ fun xs ys i => xs i + ys i) := by
  simpa only [Spec.Real, KernelIO₂.Implements, optimizedIO, floatAddIO,
    VectorAdd.optimizedIO, VectorAdd.addIO, ComputeKernel.toAlgKernel, optimized_projection]
    using VectorAdd.add_kernel_optimized_correctness B

#axiomsClean float_add_optimized_correctness
#stmtSurfaceSubset float_add_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.KernelIO₂.Implements, VeriTile.Triton.KernelIO₂.B]

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeAddCorrect
