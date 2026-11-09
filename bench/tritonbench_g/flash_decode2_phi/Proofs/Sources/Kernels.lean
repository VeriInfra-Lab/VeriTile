import bench.tritonbench_g.flash_decode2_phi.Kernels
import bench.tritonbench_g.flash_decode2_llama.Kernels

/-! Checked decomposition of the canonical source quotations; no duplicate
kernel implementation is retained here. -/
namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
noncomputable section
def kernel (masked : Bool) (H S D Q BD : Nat) : ComputeKernel :=
  if masked then phiKernel H S D Q BD else llamaKernel H S D Q BD

def prologue (H S D Q BD : Nat) : List ComputeStmt :=
  (phiKernel H S D Q BD).surfaceBody.take 10

def body (masked : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  match (kernel masked H S D Q BD).surfaceBody.getD 10 (.effectMarker "absent") with
  | ComputeStmt.forRangeInt _ _ _ _ b => b
  | _ => []

def epilogue (masked : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  (kernel masked H S D Q BD).surfaceBody.drop 11


end
end VeriTile.Bench.Repairs.FlashDecode2
