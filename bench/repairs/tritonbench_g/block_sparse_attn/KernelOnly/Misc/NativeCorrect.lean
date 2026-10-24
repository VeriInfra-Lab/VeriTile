import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Spec
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PublicCorrectness

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.FPEquiv
open VeriTile Triton FP.Structural
noncomputable section

/-- The selected FP32 source and original raw scale ABI implement the
independent ordered native recurrence. All FP32 words remain admitted;
fixed-K8 instruction semantics and replica-owner lowering are unchanged. -/
theorem kernel_native (ops : PhysicalKernel.NativePrimitives)
    (p : Parameters) (batches visits : Nat) (geometry : PublicIO.Geometry p) :
    (PublicIO.nativeIO ops p batches visits).ImplementsOn
      (PublicIO.Admissible p batches visits) (PublicIO.nativeFormula ops p batches visits) := by
  exact PublicIO.native_implements ops p batches visits geometry

#stmtSurfaceSubset kernel_native ⊆
  [PhysicalKernel.NativePrimitives,Parameters,Replicas,Word,PublicIO.Geometry,PublicIO.nativeIO,
   ProgramIO.ImplementsOn,PublicIO.Admissible,PublicIO.nativeFormula]
-- NativePrimitives is the retained local instruction/literal/scale boundary.
-- This source-to-native observation introduces no FP transformation rule or
-- additional experimental numerical assumption.
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.FPEquiv
