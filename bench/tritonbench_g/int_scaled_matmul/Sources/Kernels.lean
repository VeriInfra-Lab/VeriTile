import bench.tritonbench_g.int_scaled_matmul.Kernels
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Float.IntegerDot

/-! Checked proof decomposition of the actual repaired sources. The independent
integer specifications and execution proofs use these exact source fragments. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section
abbrev int_scaled_matmul_matmul_surface := VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource
abbrev original := VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled
abbrev source := VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled

def prelude (p : Parameters) := (source p).surfaceBody.take 12
def originalBody (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[12]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def body (p : Parameters) := originalBody p
def epilogue (p : Parameters) := (source p).surfaceBody.drop 13
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Triton FP.Structural
noncomputable section
abbrev typedScaled := VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource
abbrev original := VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled
abbrev source := VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled

def prelude (p : Parameters) := (source p).surfaceBody.take 16
def originalBody (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[16]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def body (p : Parameters) := originalBody p
def epilogue (p : Parameters) := (source p).surfaceBody.drop 17
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
