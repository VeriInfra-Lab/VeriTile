import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.FPObservations

/-! The existing store-inline transformation on the repaired kernel. The
comparison is opaque and must be implemented; no ordering law, finiteness
condition, approximation, or numerical transformation assumption is used. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- The complete public windows/profile/domain are shared in IO.lean. -/
def original := FloatingIO.original
def inlined := FloatingIO.inlined
def assumptions := FloatingIO.assumptions

/-- Successful kernel executions, output equality, and each execution's
exterior memory frame. The candidate inlines `result` into the masked store. -/
specification relu_inlined_equiv (N : Nat) :
    original N ≡[assumptions] inlined N := FPObservations.io_equiv N

#stmtSurfaceSubset relu_inlined_equiv ⊆
  [original, inlined, assumptions, Spec.Equiv, FP.Kernel₁,
   FP.Scheduled.instProgramSyntaxConditionalIO₁]
#print_fp_assumptions relu_inlined_equiv
#axiomsClean relu_inlined_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.FPEquiv
