import bench.tritonbench_g.rmsnorm_implementation.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RmsnormImplementation.Interface
open VeriTile Triton
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Equality of the complete judgments, including safety, execution and frame. -/
theorem logical_iff (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    (logicalIO candidate sxb sxm sxk sw sob som sok n eps ⊨
      fun (xs, weights) k =>
        (xs k / Real.sqrt ((∑ j : Fin n, (xs j)^2)/(n:ℝ)+eps)) * weights k) ↔
    (rowIO candidate sxb sxm sxk sw sob som sok n eps).RegionalScheduledImplements
      oneProgram (fun v _ k => formula eps (v (0 : Fin 2)) (v (1 : Fin 2)) k) := Iff.rfl

end
end VeriTile.Bench.Repairs.RmsnormImplementation.Interface
