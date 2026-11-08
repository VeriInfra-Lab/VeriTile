import bench.tritonbench_g.seeded_dropout.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.SeededDropout.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open VeriTile.Bench.TritonBenchG.SeededDropout
noncomputable section

/-- Selected source, canonical Philox decoder and the original masked input/output
windows. Real arithmetic does not assert native FP32 exactness. -/
abbrev dropoutIO (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) :=
  KernelPublic.logicalIO Whole.realAlgebra k X O n p seed pid

specification seeded_dropout_kernel_correctness (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat)
    (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed) :
    VeriTile.Spec.Real ((dropoutIO k X O n p seed pid) ⊨
      fun xs i =>
        if Random.randReal (seedValue k seed) ((pid*1024+i.val : Nat) : Int) .bits32 10 > p
        then xs i/(1-p) else 0) :=
  KernelPublic.logical_real k X O n p seed pid hn hp hs

#specNonCircular Whole.realFormula avoiding [kernel,selected,sourceFamily,indexCommutedKernel,selected,sourceFamily,KernelExecution.originalExecution]
#axiomsClean seeded_dropout_kernel_correctness
#stmtSurfaceSubset seeded_dropout_kernel_correctness ⊆
  [SeedKind,RegionName,SeedCarrier,gridCount,SeedABI,VeriTile.Spec.Real,
   dropoutIO,LogicalProgramIO.Implements,
   Random.randReal,Random.OffsetWidth.bits32,seedValue]
end
end VeriTile.Bench.Repairs.SeededDropout.Correct
