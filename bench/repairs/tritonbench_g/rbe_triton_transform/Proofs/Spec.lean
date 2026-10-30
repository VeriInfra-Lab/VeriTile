import bench.repairs.tritonbench_g.rbe_triton_transform.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RbeTritonTransform
open VeriTile.Triton
noncomputable section

namespace Spec
/-- Input-only logical rotation; a missing odd partner is zero. -/
def angle (r : Request) (m n : Nat) : ℝ :=
  ((r.position : ℝ)+(m : ℝ)) / Real.rpow 10000 ((n%128 : Nat)/ (128 : ℝ))

def pair (r : Request) (x : Nat → ℝ) (b m n : Nat) (imag : Bool) : ℝ :=
  let u := x (r.offset b m n)
  let v := if n+1<r.columns then x (r.offset b m (n+1)) else 0
  let a := angle r m n
  if imag then u*Real.sin a+v*Real.cos a else u*Real.cos a-v*Real.sin a

def result (r : Request) (x : Nat → ℝ) (b m c : Nat) : ℝ :=
  pair r x b m (c/2*2) (decide (c%2=1))

#specNonCircular result avoiding [FP.Structural.exec]
end Spec
end
end VeriTile.Bench.Repairs.RbeTritonTransform
