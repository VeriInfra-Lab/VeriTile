import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Memory.KernelSpec.Basic
import bench.examples.LogExp.Kernels
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.Source

/-! The logical and floating-point program interfaces. -/

noncomputable section

namespace VeriTile.Bench.Examples.LogExp.Correct

open Triton

open scoped VeriTile.Triton.KernelIO₁

set_option maxHeartbeats 1600000

/-- The original source, under the real interpretation of KernelIO₁. -/
def originalIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ where
  kernel := originalKernel xReg yReg B
  inp := xReg
  out := yReg
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The piecewise candidate shares its source with the FP proof. -/
def optimizedIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ :=
  { originalIO xReg yReg B with kernel := optimizedKernel xReg yReg B, projection := by rfl }

end VeriTile.Bench.Examples.LogExp.Correct

noncomputable section

namespace VeriTile.Bench.Examples.LogExp.FPEquiv

open Triton FP FP.Structural FP.Guarded FP.LogExp

open scoped VeriTile.Spec

set_option maxHeartbeats 1600000

def typedIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ where
  kernel := originalKernel xReg yReg B
  inp := xReg
  out := yReg
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

end VeriTile.Bench.Examples.LogExp.FPEquiv
