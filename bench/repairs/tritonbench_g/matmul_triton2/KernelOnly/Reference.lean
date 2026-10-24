import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.RealSum

/-! Independent exact matrix product over the complete K range. -/
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Reference
open VeriTile.Triton Revision001
noncomputable section

def real (K : Nat) (a b : Nat → Nat → ℝ) (r c : Nat) : ℝ :=
  RealSum.sum (fun k => a r k*b k c) K

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Reference
