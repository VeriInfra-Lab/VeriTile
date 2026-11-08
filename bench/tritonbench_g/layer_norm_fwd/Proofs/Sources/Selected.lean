import bench.tritonbench_g.layer_norm_fwd.Kernels

namespace VeriTile.Bench.Repairs.LayerNormFwdSelectedSource
open VeriTile.Triton
def rawOriginal (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
          (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
          (stride_x_row stride_y_row stride_res_row stride_res_out_row
            stride_x1_row stride_y1_row M N : Nat)
          (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
          (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
          (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
            STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) : ComputeKernel :=
          originalKernel X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE SEEDS DROPOUT_MASK Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row stride_x1_row stride_y1_row M N eps dropout_p IS_RMS_NORM BLOCK_N hBlock HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1

def rawIndexCommuted (X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE : RegionName)
          (SEEDS : Region .int) (DROPOUT_MASK : Region .bool) (Mean Rstd : RegionName)
          (stride_x_row stride_y_row stride_res_row stride_res_out_row
            stride_x1_row stride_y1_row M N : Nat)
          (eps dropout_p : ℝ) (IS_RMS_NORM : Bool)
          (BLOCK_N : Nat) (hBlock : Random.ArangeSizeDomain BLOCK_N)
          (HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT
            STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1 : Bool) : ComputeKernel :=
          indexCommutedKernel X Y W B RESIDUAL X1 W1 B1 Y1 RESIDUAL_OUT ROWSCALE SEEDS DROPOUT_MASK Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row stride_x1_row stride_y1_row M N eps dropout_p IS_RMS_NORM BLOCK_N hBlock HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS HAS_DROPOUT STORE_DROPOUT_MASK HAS_ROWSCALE HAS_X1 HAS_W1 HAS_B1

end VeriTile.Bench.Repairs.LayerNormFwdSelectedSource
