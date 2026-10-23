import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Contract
open VeriTile.Triton
open VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option pp.proofs false
set_option pp.universes false
set_option maxRecDepth 20000
#reduce (ScalarCast.TypedSource.kernel "A" "B" "C" "bias" "state_x" "state_w"
  17 33 65 65 1 33 1 33 1 16 32 32 8 2 2 Bool.false Bool.true).surfaceBody[30]?
