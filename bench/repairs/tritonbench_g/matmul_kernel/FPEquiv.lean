import bench.repairs.tritonbench_g.matmul_kernel.Proofs.PairIO

namespace VeriTile.Bench.Repairs.MatmulKernel.FPEquiv
open VeriTile.Triton FP.Structural KernelSource
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

def original (p : Config) (M : Algebra α) (D : DotAcc α) :=
  ProgramIO.Implementation.original (PublicIO.io p M D) (fun _ => True)
def remainingK (p : Config) (M : Algebra α) (D : DotAcc α) :=
  ProgramIO.Implementation.ofVariant (PairIO.variant p M D) (fun _ => True)

/-- Strength-reducing both K-tail masks preserves the complete two-dimensional
source grid under the original input binding and IO windows. Both actual runs
are safe and successful, all complete output cells agree, and both exterior
frames hold. All FP32 loads/live dots, FP16 narrowing and FP32 storage widening
are unchanged; the rewrite needs only exact discrete arithmetic. -/
specification source_pair_equiv (p : Config) (valid : p.Valid) (M : Algebra α) (D : DotAcc α) :
    original p M D ≡[[]] remainingK p M D :=
  ProgramIO.Implementation.of_equivalent (PairIO.variant p M D) (PairIO.equivalent p valid M D)

#print_fp_assumptions source_pair_equiv
#axiomsClean source_pair_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulKernel.FPEquiv
