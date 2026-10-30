import bench.tritonbench_g.quant_transpose_kernel.Proofs.ByteIO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.FPEquiv
open VeriTile Triton FP.Structural KernelModels
open scoped VeriTile.Spec
noncomputable section

/-- Both actual complete grouped schedules preserve the same INT8 bytes and
exact raw exterior frames. Only an integer input-row product commutes; all
source-ordered FP32 operations and the pinned signed64 conversion are retained. -/
specification kernel_native_equiv (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : KernelSource.Parameters) (separated : KernelGrid.Separate p) :
    IO.originalProgram multiply rest p ≡[[]] IO.commutedProgram multiply rest p :=
  IntegerProgramIO.ObservedImplementation.equiv_of_equivalent
    (IO.commuted (nativeAlgebra multiply rest) nativeRound p)
    (ByteIO.equivalent (nativeAlgebra multiply rest) nativeRound p separated)

#print_fp_assumptions kernel_native_equiv
#axiomsClean kernel_native_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.FPEquiv
