import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.Source
import bench.examples.LogExp.Kernels
import bench.examples.LogExp.Proofs.FP

/-! The Real and FP entries expose the existing mathematical input and finite row requirements. -/

noncomputable section

namespace VeriTile.Bench.Examples.LogExp.FPEquiv

open Triton FP FP.Structural FP.Guarded FP.LogExp

open scoped VeriTile.Spec

set_option maxHeartbeats 1600000

/-- The candidate prefix reads `a` and evaluates the branch selector. Successful
execution requires comparison support, for either branch outcome. The only
numerical input restriction is that each loaded `a` is finite. -/
def domain (xReg : RegionName) (B : Nat) : Precondition where
  code := Source.prefixBefore (optimizedKernel xReg "y" B).surfaceBody "fallback_a"
  guards := [⟨"a", [B], .finite⟩]

end VeriTile.Bench.Examples.LogExp.FPEquiv
