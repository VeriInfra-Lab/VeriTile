import bench.tritonbench_g.token_attn_mistral.Proofs.AffineEquivalence

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.FPEquiv
open VeriTile Triton FP.Structural TokenAttnMistral
open scoped VeriTile.Spec
noncomputable section

/-- Original complete source schedule, on the original logical input domain. -/
def original (R : Native.Reductions) (p : Parameters) (work : List Grid.PID)
    (counts : Fin work.length → Nat) :=
  ProgramIO.Implementation.original (IO.native R p work counts) (Ports.Domain p work counts)

/-- Complete FP32 candidate computes the uniform affine probability origin
scalarly, avoiding one vector addition in address formation. -/
def affineAddress (R : Native.Reductions) (p : Parameters) (work : List Grid.PID)
    (counts : Fin work.length → Nat) :=
  ProgramIO.Implementation.ofVariant
    (IO.affine Native.algebra (Native.reduction R) p work counts)
    (Ports.Domain p work counts)

specification affine_address_equiv (R : Native.Reductions) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    original R p work counts ≡[([] : Spec.Assumptions _)] affineAddress R p work counts :=
  ProgramIO.Implementation.of_equivalentOn _
    (AffineEquivalence.equivalent Native.algebra (Native.reduction R) p work counts separate unique)

#axiomsClean affine_address_equiv
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.FPEquiv
