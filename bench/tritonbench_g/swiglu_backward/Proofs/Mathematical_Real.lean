import bench.tritonbench_g.swiglu_backward.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluBackward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₃ₓ₃

def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val

/-! ### The `⊨` specification -/

end VeriTile.Bench.TritonBenchG.SwigluBackward
