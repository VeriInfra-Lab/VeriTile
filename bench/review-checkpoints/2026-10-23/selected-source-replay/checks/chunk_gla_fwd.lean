import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.chunk_gla_fwd.Contract
import bench.repairs.tritonbench_g.chunk_gla_fwd.Correct
import bench.repairs.tritonbench_g.chunk_gla_fwd.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.Correct.producer_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.Correct.consumer_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv.inter_equiv
#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv.intra_equiv
#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv.split_equiv
#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv.merge_equiv
#axiomsClean VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv.output_equiv
