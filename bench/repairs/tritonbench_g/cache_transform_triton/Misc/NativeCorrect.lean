import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.IOResult

/-! Exact native byte-copy IO for the single selected source. Every copied
byte, signed row index, full launch and exterior frame is retained. There is no
floating arithmetic transformation or duplicate implementation to equate.
The byte representation preserves the FP32 grouping supplied by the inputs.
-/
namespace VeriTile.Bench.Repairs.CacheTransform.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

theorem kernel_native (M : Algebra Unit) (p : Parameters) (n capacity : Nat)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit) :
    (cacheIO M p n capacity).ImplementsOnCells
      (CacheIO.Domain M p n capacity) (CacheIO.formula M p n capacity) :=
  IOProof.implements M p start block grid n capacity

#stmtSurfaceSubset kernel_native ⊆
  [FP.Structural.Algebra, Parameters, Parameters.start, Parameters.block, Parameters.count,
   blocks, gridLimit, cacheIO, CacheIO.Domain, CacheIO.formula, ProgramIO.ImplementsOnCells]
#axiomsClean kernel_native
#specNonCircular byteTarget avoiding [source,KernelSource.launch,gather,cacheIO]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.FPEquiv
