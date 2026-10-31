import bench.tritonbench_g.swiglu_fwd.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Lane `i`'s address in a row-strided buffer for program
`(row, col_block) = (pids 0, pids 1)`: row base `pids 0 * stride` plus column
`pids 1 * BLOCK_N + i`. -/
def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val

end VeriTile.Bench.TritonBenchG.SwigluFwd
