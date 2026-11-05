import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.Reciprocal
import bench.examples.FloatDTypeSoftmax.Kernels
import bench.examples.FloatDTypeSoftmax.Proofs.FP

/-! The Real theorem uses a positive tile size. The FP division/reciprocal comparison uses the row domain declared below. -/

namespace VeriTile.Bench.Examples.FloatDTypeSoftmaxFPEquiv

open VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels

open VeriTile Triton

open FP.Structural FP.Guarded

set_option maxHeartbeats 1600000

/-- Domain requirements for the two operands at each scalar rewrite site.
No distribution or experimental shape is imposed on the symbolic kernel. -/
def domain (B : Nat) : Precondition where
  code :=  (commonPrefix "x" B).surfaceBody
  guards := [⟨"e", [B], .finite⟩, ⟨"s", [], .finite⟩, ⟨"s", [], .nonzero⟩]

abbrev Rules := FP.Reciprocal.Rules .fp64_fp32

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.FloatDTypeSoftmaxFPEquiv
