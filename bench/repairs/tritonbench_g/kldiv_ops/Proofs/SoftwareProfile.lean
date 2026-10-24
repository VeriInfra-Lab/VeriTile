import VeriTile.Triton.Float.ScalarBackend
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Library
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Tile

namespace VeriTile.Bench.Repairs.KldivOps.FPProfile
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := FP.Value .fp32

def config : FP.Config := .ieeeValue

/-- RN-even software add/sub/mul, identity FP32 loads/casts and sign negation.
The remaining fields used by this task are exactly max, tl.log, tl.exp and
per-block reduceSum; their native local interpretations remain explicit. -/
def algebra (rest : Algebra Word) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest config
  {base with literal:=fun p d x=>if p=some .fp32 ∧ d=.real ∧ x = -1
    then FP.Value.ofBits .fp32 0xbf800000 else base.literal p d x}

/-- Torch's dimension-0 and global sums and scalar-int division are separate
local primitive ports. Their order/conversion cannot be inferred from Triton. -/
def numerics (sum : Bool→List Word→Word) (divide : Word→Nat→Word) : Host.Numerics Word where
  zero := FP.Value.zero .fp32
  one := FP.Value.ofBits .fp32 0x3f800000
  neg := FP.Value.neg
  equal := FP.eq config
  sum := sum
  divide := divide
  multiply := FP.mul config

/-- Exact scalar word graph, including separate subtraction and multiplication.
There is no FMA substitution at this source site. -/
theorem scalar_default (rest : Algebra Word) (eps y target : Word) :
    Tile.scalar (algebra rest) Bool.false eps y target=
      FP.mul config target (FP.sub config
        (rest.unary (some .fp32) .log (rest.binary (some .fp32) .real .max target eps)) y) := by
  simp [Tile.scalar,algebra,FP.Software.scalarFP32,FP.Software.fp32Context]

theorem scalar_log (rest : Algebra Word) (eps y target : Word) :
    Tile.scalar (algebra rest) Bool.true eps y target=
      FP.mul config (rest.unary (some .fp32) .exp target) (FP.sub config target y) := by
  simp [Tile.scalar,algebra,FP.Software.scalarFP32,FP.Software.fp32Context]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.FPProfile
