import bench.tritonbench_g.seeded_dropout.Kernels
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- The complete algorithm projection retains source seed profiles, all
statements, canonical random generation and eager real division. -/
def reference (seedKind : Random.IntegerType) (swapped : Bool)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier seedKind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) : Kernel :=
  (source .fp64 seedKind swapped x y N p seed B hB).eraseDType.toAlgKernel

theorem kernel (storage : Storage) (seedKind : Random.IntegerType) (swapped : Bool)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier seedKind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    (source storage seedKind swapped x y N p seed B hB).eraseDType.toAlgKernel =
      reference seedKind swapped x y N p seed B hB := by
  cases storage <;> cases seedKind <;> cases swapped
  all_goals simp only [reference, source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    Except.bind, bind]
  all_goals simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.SeededDropout.Erasure
