import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.AffineEquivalence
import bench.repairs.tritonbench_g.token_attn_reduceV.Native

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.FPEquiv
open VeriTile Triton FP.Structural TokenAttnReduceV
open scoped VeriTile.Spec
noncomputable section

def original (R : Native.Reductions) (p : Parameters) (B H : Nat)
    (counts : Fin B → Nat) (work : List Grid.PID) :=
  ProgramIO.Implementation.original
    (Interface.programIO Native.algebra (Native.reduction R) p B H counts work)
    (Interface.LengthsMatch p B H counts)

def affineAddress (R : Native.Reductions) (p : Parameters) (B H : Nat)
    (counts : Fin B → Nat) (work : List Grid.PID) :=
  ProgramIO.Implementation.ofVariant
    (Interface.affine Native.algebra (Native.reduction R) p B H counts work)
    (Interface.LengthsMatch p B H counts)

/-- Actual complete gathered-PV programs with a scalar affine probability origin.
Every source FP32 operation and padded lane remains in its original order. -/
specification affine_address_equiv (R : Native.Reductions) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (work : List Grid.PID)
    (order : (Grid.fullGrid B H).Perm work)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    original R p B H counts work ≡[([] : Spec.Assumptions _)] affineAddress R p B H counts work :=
  ProgramIO.Implementation.of_equivalentOn _
    (AffineEquivalence.equivalent Native.algebra (Native.reduction R) p B H counts work order separate unique)

#axiomsClean affine_address_equiv
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.FPEquiv
