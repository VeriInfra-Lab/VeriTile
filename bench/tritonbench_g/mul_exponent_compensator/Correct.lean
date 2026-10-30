import bench.tritonbench_g.mul_exponent_compensator.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Correct
open VeriTile Triton FP.Structural Interface
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The complete original arbitrary-order grid, with logical vector inputs
and outputs. Tail, empty and in-place cases retain the original memory judgment. -/
def realIO (src dst : RegionName) (n B : Nat) (order : List Nat) :=
  logicalIO KernelArithmetic.realAlgebra src dst n B order
specification kernel_real (src dst : RegionName) (n B : Nat) (positive : 0<B)
    (order : List Nat) (schedule : order.Perm (List.range (grid n B))) :
    Spec.Real (realIO src dst n B order ⊨ fun xs i => xs i * (2 : ℝ)^112) :=
  LogicalIOProof.implements src dst n B positive order schedule

#stmtSurfaceSubset kernel_real ⊆
  [Spec.Real,RegionName,grid,realIO,LogicalProgramIO.Implements]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Correct
