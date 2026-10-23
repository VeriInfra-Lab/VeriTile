import bench.repairs.tritonbench_g.matrix_vector_multip.FPEquiv

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
open scoped VeriTile.Spec
noncomputable section

def original (native : Scalars) (N M BN BM pid : Nat) (g : Graph BN BM) :=
  ProgramIO.Implementation.original (ExplicitFMA.originalIO native N M BN BM pid g) (fun _ => True)

def explicitFused (native : Scalars) (N M BN BM pid : Nat) (g : Graph BN BM) :=
  ProgramIO.Implementation.ofVariant (ExplicitFMA.variant native N M BN BM pid g) (fun _ => True)

/-- The complete looping source versus an explicit-FMA implementation. The
selected source's existing native lowering already uses one ternary FMA for
its carry; both implementations retain that exact graph and reduction order.
All input words, zero extents, masked tails and complete column loops remain.
This is not a split-multiply/add-to-FMA equality or a hardware guarantee. -/
theorem explicit_fma_equiv (native : Scalars) (N M BN BM pid : Nat)
    (g : Graph BN BM) (width : 0<BM) :
    original native N M BN BM pid g ≡[[]] explicitFused native N M BN BM pid g :=
  ProgramIO.Implementation.of_equivalent (ExplicitFMA.variant native N M BN BM pid g)
    (ExplicitFMA.equivalent native N M BN BM pid g width)

#print_fp_assumptions explicit_fma_equiv
#axiomsClean explicit_fma_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv
