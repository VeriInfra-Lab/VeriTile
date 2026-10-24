import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.layer_norm_triton.Contract
import bench.repairs.tritonbench_g.layer_norm_triton.Correct
import bench.repairs.tritonbench_g.layer_norm_triton.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.LayerNormTriton.Correctness.forward_correct
#axiomsClean VeriTile.Bench.Repairs.LayerNormTriton.Correctness.grouped_correct
#axiomsClean VeriTile.Bench.Repairs.LayerNormTriton.Correctness.final_correct
#axiomsClean VeriTile.Bench.Repairs.LayerNormTriton.Equivalence.forward_equiv
#axiomsClean VeriTile.Bench.Repairs.LayerNormTriton.Equivalence.grouped_equiv
#axiomsClean VeriTile.Bench.Repairs.LayerNormTriton.Equivalence.final_equiv
