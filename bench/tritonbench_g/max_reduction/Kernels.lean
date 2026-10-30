import bench.tritonbench_g.max_reduction.Sources.Mathematical_Kernels

/-! The FP32 source selections used by both public proof entries. The three
readable Triton quotations are shared transparently with the original task;
there is no second handwritten AST. Python's explicit int64 index
widening is connected to these mathematical indices by Proofs/Machine.lean.
Only integer index multiplication or mask syntax changes in the retained
candidates; floating reductions and their padding stay in the same order. -/
namespace VeriTile.Bench.Repairs.MaxReduction
open VeriTile.Triton

/-- `max_kernel_1`: masked FP32 load, max, and one FP32 store. -/
def stage1 := VeriTile.Bench.TritonBenchG.MaxReduction.stage1 .fp32 Bool.false
/-- The existing candidate commutes `pid * BLOCK_SIZE`. -/
def stage1Commuted := VeriTile.Bench.TritonBenchG.MaxReduction.stage1 .fp32 Bool.true

/-- `max_kernel_2`: masked FP32 intermediate load and final maximum. -/
def stage2 := VeriTile.Bench.TritonBenchG.MaxReduction.stage2 .fp32 Bool.false
/-- The existing candidate uses `not (mid_size <= offset)`. -/
def stage2Complement := VeriTile.Bench.TritonBenchG.MaxReduction.stage2 .fp32 Bool.true

/-- `max_kernel`: FP32 values and signed-int64 indices, with both masks. -/
def dimension := VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource .fp32 Bool.false
/-- The existing candidate commutes `pid_m * BLOCK_M`. -/
def dimensionCommuted := VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource .fp32 Bool.true

end VeriTile.Bench.Repairs.MaxReduction
