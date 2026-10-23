import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Kernels
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Spec
import bench.repairs.tritonbench_g.chunk_retention.Proofs.PublicCorrectness

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section

/-- Actual FP32 source observation agrees with the independent ordered scalar
IEEE graph. All FP32 input words remain admitted; numeric ports retain their
original local instruction/literal meaning and compiler materialization scope. -/
theorem source_fp_equiv (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32)
    (literals : KernelStaging.LiteralLaws (wordOps (Numeric.nativeISA ports)) literal)
    (a : Call) (p grid : Nat→Nat) (legal : PublicIO.Legal a p grid) :
    (PublicIO.nativeIO (wordOps (Numeric.nativeISA ports)) literal a p grid).Implements
      (PublicIO.nativeFormula (wordOps (Numeric.referenceISA u)) literal a p) := by
  exact PublicIO.fp_implements u ports laws literal literals a p grid legal

#stmtSurfaceSubset source_fp_equiv ⊆
  [Numeric.UnaryModels,Numeric.ScalarPorts,Numeric.ScalarLaws,KernelStaging.LiteralLaws,
   wordOps,Numeric.nativeISA,Numeric.referenceISA,Call,KernelPacket.Call,PublicIO.Legal,
   PublicIO.nativeIO,PublicIO.nativeFormula,AddressedProgramIO.Implements,
   KernelStaging.Term,KernelStaging.instInhabitedTerm]
-- ScalarLaws and LiteralLaws are explicit primitive interpretation boundaries.
-- This is one source observed through its native graph; no FP rewrite rule or
-- additional experimental numerical assumption is introduced.
#axiomsClean source_fp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
