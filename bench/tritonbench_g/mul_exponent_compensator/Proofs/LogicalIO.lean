import bench.tritonbench_g.mul_exponent_compensator.IO
import bench.tritonbench_g.mul_exponent_compensator.Proofs.IOComplete

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.LogicalIOProof
open VeriTile.Triton FP.Structural Interface
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem headline_iff (src dst : RegionName) (n B : Nat) (order : List Nat) :
    (logicalIO KernelArithmetic.realAlgebra src dst n B order ⊨
      fun xs i => xs i * (2 : ℝ)^112) ↔
    (io KernelArithmetic.realAlgebra src dst n B order).Implements
      (formula KernelArithmetic.mathematical src dst n) := by
  have target : (logicalIO KernelArithmetic.realAlgebra src dst n B order).lift
      (fun xs i => xs i * (2 : ℝ)^112) = formula KernelArithmetic.mathematical src dst n := by
    funext xs port i
    fin_cases port
    rfl
  change (io _ _ _ _ _ _).Implements _ ↔ _
  rw [target]

theorem implements (src dst : RegionName) (n B : Nat) (positive : 0<B)
    (order : List Nat) (schedule : order.Perm (List.range (grid n B))) :
    (logicalIO KernelArithmetic.realAlgebra src dst n B order ⊨
      fun xs i => xs i * (2 : ℝ)^112) :=
  (headline_iff src dst n B order).2 (IOComplete.real_implements src dst n B positive order schedule)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.LogicalIOProof
