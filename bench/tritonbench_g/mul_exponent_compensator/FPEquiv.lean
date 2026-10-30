import bench.tritonbench_g.mul_exponent_compensator.Proofs.Commuted

/-! The existing scalar-left multiplication transformation on the complete
masked FP32 source and every full program order. The concrete software model
proves commutation for every word, including NaNs, infinities and signed zeros. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPEquiv
open VeriTile.Triton FP.Structural Interface
open scoped VeriTile.Spec
noncomputable section

def original (profile : FP.Config) (src dst : RegionName) (n B : Nat)
    (order : List Nat) :=
  ProgramIO.Implementation.original
    (io (KernelArithmetic.nativeAlgebra profile) src dst n B order) (fun _ => True)

def scalarLeft (profile : FP.Config) (src dst : RegionName) (n B : Nat)
    (order : List Nat) :=
  ProgramIO.Implementation.ofVariant
    (Commuted.variant (KernelArithmetic.nativeAlgebra profile) src dst n B order) (fun _ => True)

/-- Both actual complete grids are safe, succeed, produce the same typed cells
and preserve the exterior frame. Empty/tail/in-place cases and all supported
blocks are retained, with no finiteness premise or experimental rule. -/
specification scalar_left_equiv (profile : FP.Config) (src dst : RegionName)
    (n B : Nat) (positive : 0<B) (order : List Nat)
    (schedule : order.Perm (List.range (grid n B))) :
    original profile src dst n B order ≡[[]] scalarLeft profile src dst n B order := by
  apply ProgramIO.Implementation.of_equivalent
    (Commuted.variant (KernelArithmetic.nativeAlgebra profile) src dst n B order)
  exact ProgramIO.EquivalentOn.of_implementsOn
    (IOComplete.native_implements profile src dst n B positive order schedule)
    (Commuted.implements profile src dst n B positive order schedule)

#print_fp_assumptions scalar_left_equiv
#axiomsClean scalar_left_equiv
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPEquiv
