import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.RmsnormFused.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra)
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Independent mathematical RMSNorm on logical input arrays. -/
def kernel_formula (N : Nat) (epsilon : ℝ) (xs : Nat → Nat → ℝ)
    (weights : Nat → ℝ) (row col : Nat) : ℝ :=
  xs row col * (Real.sqrt ((∑ j ∈ Finset.range N,(xs row j)^2)/(N:ℝ)+epsilon))⁻¹ *
    weights col

/-- IO.lean declares full X/W input spans, the full Y output span, no scratch,
and every source invocation with its row PID and runtime epsilon register. -/
def rmsIO (shape : Shape) (epsilon : ℝ) (rows : List Nat) : ProgramIO ℝ ℝ :=
  Interface.programIO realAlgebra shape epsilon rows id (fun _ => id)

/-- Every row runs the complete canonical FP32 source. The shared IO judgment
includes actual source access safety, all logical outputs and the exterior frame. -/
specification kernel_real (shape : Shape) (valid : shape.Valid) (epsilon : ℝ)
    (rows : List Nat) (all : rows.Perm shape.grid) :
    Spec.Real (Interface.logicalIO shape epsilon rows ⊨ fun values row col =>
      let xs := values.1
      let weights := values.2
      xs row col * (Real.sqrt
        ((∑ j : Fin shape.width, (xs row j)^2)/(shape.width:ℝ)+epsilon))⁻¹ * weights col) :=
  (Interface.logical_iff shape epsilon rows).mpr (Ideal.io_correct shape valid epsilon rows all)

#stmtSurfaceSubset kernel_real ⊆
  [Spec.Real, Shape, Shape.Valid, Shape.grid, Shape.width, Shape.rows,
   Interface.logicalIO, LogicalProgramIO.Implements]
#specNonCircular kernel_formula avoiding [source,Runtime.value,Runtime.run,FP.Structural.exec]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Correct
