import bench.tritonbench_g.chunk_gla_simple.Proofs.Pair

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPEquiv
open VeriTile Triton FP.Structural Host
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 16000000

abbrev original := PairIO.original
abbrev rowReused := PairIO.rowReused

/-- Compare complete selected min-BK source grids on the original arbitrary
symbolic inputs. Reuse the already generated lane vector for row addresses;
retain every FP32 dot, cast, gate and scale operation in its original order.
Both actual schedules are safe and succeed, with all typed output cells and
complete exterior memory frames equal. No numerical rewrite assumption. -/
specification whole_kernel_equiv (c : Shape) (bt : 0<c.BT) :
    original c ≡[[]] rowReused c := PairIO.main c bt

#axiomsClean whole_kernel_equiv
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPEquiv
