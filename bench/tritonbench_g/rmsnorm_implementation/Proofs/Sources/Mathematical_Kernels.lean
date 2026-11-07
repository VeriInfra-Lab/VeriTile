import bench.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Triton.Source.RealCastView

/-! Precision-erased mathematical view and compatibility profile adapters.
Every program delegates to a complete raw body in Kernels.lean. The explicit
Real identity cast at xf is a proof view, not another implementation. -/
namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton

def rmsnorm_implementation
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) : ComputeKernel :=
  RealCastView.ofKernel "xf"
    (VeriTile.Bench.Repairs.RmsnormImplementation.kernel x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps).eraseDType.toAlgKernel

/-- Every retained storage combination selects the same original Python body. -/
def originalTyped (input weight output : Storage) : Source :=
  VeriTile.Bench.Repairs.RmsnormImplementation.originalForKernel input weight output

/-- Select the complete existing address candidate without statement surgery. -/
def typedKernel (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  VeriTile.Bench.Repairs.RmsnormImplementation.family input weight output v
    X W O sxb sxm sxk sw sob som sok N B eps
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
