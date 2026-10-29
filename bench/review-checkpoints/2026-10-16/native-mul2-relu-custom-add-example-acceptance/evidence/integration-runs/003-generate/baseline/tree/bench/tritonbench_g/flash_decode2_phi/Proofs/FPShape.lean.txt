import bench.tritonbench_g.flash_decode2_phi.Proofs.FPSource
import bench.tritonbench_g.flash_decode2_phi.FlashDecode2Phi

/-! All four source-stage projections retain the original full parameter
lists, masks, load/store addresses and operation order. -/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
open VeriTile.Triton

theorem normalization_store_kernel_projection
    (Acc SumExp Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    (source_normalization_store_kernel Acc SumExp Out head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h stride_obs stride_oh stride_od BLOCK_DMODEL).toAlgKernel.body =
      (flash_decode2_phi_normalization_store_kernel Acc SumExp Out head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h stride_obs stride_oh stride_od BLOCK_DMODEL).toAlgKernel.body := by
  rfl

#axiomsClean normalization_store_kernel_projection

theorem accumulator_step_kernel_projection
    (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut : RegionName)
    (block_seq_n
      head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h
      BLOCK_DMODEL : Nat) :
    (source_accumulator_step_kernel Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb stride_mid_o_eh stride_acc_b stride_acc_h stride_acc_d stride_logic_b stride_logic_h BLOCK_DMODEL).toAlgKernel.body =
      (flash_decode2_phi_accumulator_step_kernel Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb stride_mid_o_eh stride_acc_b stride_acc_h stride_acc_d stride_logic_b stride_logic_h BLOCK_DMODEL).toAlgKernel.body := by
  rfl

#axiomsClean accumulator_step_kernel_projection

theorem sum_exp_step_kernel_projection
    (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h
      stride_logic_b stride_logic_h : Nat) :
    (source_sum_exp_step_kernel Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut block_seq_n stride_mid_o_eb stride_mid_o_eh stride_sum_b stride_sum_h stride_logic_b stride_logic_h).toAlgKernel.body =
      (flash_decode2_phi_sum_exp_step_kernel Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut block_seq_n stride_mid_o_eb stride_mid_o_eh stride_sum_b stride_sum_h stride_logic_b stride_logic_h).toAlgKernel.body := by
  rfl

#axiomsClean sum_exp_step_kernel_projection

theorem final_store_slice_projection
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    (source_final_store_slice Final Out head_dim stride_final_b stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL).toAlgKernel.body =
      (flash_decode2_phi_final_store_slice Final Out head_dim stride_final_b stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL).toAlgKernel.body := by
  rfl

#axiomsClean final_store_slice_projection

end VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
