import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.Reciprocal
import bench.examples.SoftmaxReciprocal.Kernels
import bench.examples.SoftmaxReciprocal.Proofs.FP

/-! The Real theorem takes positive row size. The FP comparison uses the existing division/reciprocal domain below. -/

namespace VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv

open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels

open VeriTile Triton

open FP.Structural FP.Guarded

set_option maxHeartbeats 1600000

/-- Domain requirements for the two operands at each scalar rewrite site.
No distribution or experimental shape is imposed on the symbolic kernel. -/
def domain (B : Nat) : Precondition where
  code :=  (commonPrefix "x" B).surfaceBody
  guards := [⟨"e", [B], .finite⟩, ⟨"s", [], .finite⟩, ⟨"s", [], .nonzero⟩]

abbrev Rules := FP.Reciprocal.Rules .fp32

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
