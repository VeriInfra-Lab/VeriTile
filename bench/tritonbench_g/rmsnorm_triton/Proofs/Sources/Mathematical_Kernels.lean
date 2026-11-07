import bench.tritonbench_g.rmsnorm_triton.Kernels
import bench.tritonbench_g.rmsnorm_triton.Proofs.Sources.RealSquareView

/-! Mathematical views and profile adapters of the complete raw source.
The Real-only square view is connected by full variance-loop proofs; the FP
programs retain the actual integer-power intrinsic. -/
namespace VeriTile.Bench.TritonBenchG.RmsnormTriton
open VeriTile.Triton
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) : ComputeKernel :=
  RealSquareView.ofKernel
    (VeriTile.Bench.Repairs.RmsnormTriton.kernel x_ptr rms_w_ptr output_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps).eraseDType.toAlgKernel


def originalTyped (input weight output : Storage) : Source :=
  VeriTile.Bench.Repairs.RmsnormTriton.originalForKernel input weight output

def typedKernel (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  VeriTile.Bench.Repairs.RmsnormTriton.family input weight output v
    X W O sxb sxm sxk sw sob som sok N B eps
end VeriTile.Bench.TritonBenchG.RmsnormTriton
