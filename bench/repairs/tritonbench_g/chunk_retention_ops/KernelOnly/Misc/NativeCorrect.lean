import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOResults

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry records the remaining comparison blocker. -/

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.FPEquiv
open VeriTile Triton FP.Structural
open KernelPacket (Call)
open KernelNative
noncomputable section

/-- All four actual FP32 kernels implement the independent scalar operation
formulas under the existing primitive and literal boundaries. Every original
arbitrary PID is admitted. The shared judgment includes actual-source access
safety, successful execution, all finite outputs and exact exterior frame.
This is native observation; it does not assert an optimization or native=Real. -/
theorem kernel_native (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : S.LiteralLaws (sharedOps (wordOps (Numeric.referenceISA u))) literal) (a : Call) :
    (PublicIO.nativeIO (Numeric.nativeISA ports) literal a).Implements
      (PublicIO.nativeFormula (Numeric.referenceISA u) literal a) :=
  PublicIO.native_boundary u ports laws literal constants a

/- No floating transformation laws are introduced. ScalarLaws and LiteralLaws
are the retained primitive/model boundaries, recorded in the numerical profile. -/
#stmtSurfaceSubset kernel_native ⊆
  [Numeric.UnaryModels,Numeric.ScalarPorts,Numeric.ScalarLaws,Numeric.nativeISA,Numeric.referenceISA,
   Word,Term,VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.instInhabitedTerm,
   Call,sharedOps,wordOps,S.LiteralLaws,PublicIO.nativeIO,PublicIO.nativeFormula,
   AddressedProgramIO.Implements]
#specNonCircular PublicIO.nativeFormula avoiding [FP.Structural.exec,run,KernelPacket.reference,KernelPacket.source]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.FPEquiv
