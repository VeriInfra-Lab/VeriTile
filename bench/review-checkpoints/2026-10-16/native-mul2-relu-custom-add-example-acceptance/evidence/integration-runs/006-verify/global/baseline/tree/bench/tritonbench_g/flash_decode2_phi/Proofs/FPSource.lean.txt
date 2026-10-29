import bench.tritonbench_g.flash_decode2_llama.Proofs.FPContract

/-! Typed fp32 versions of all previously accepted Phi stage cuts.
Every source address, scalar input and mask is retained. The supplied old/new
maximum remains a stage input; no complete-loop claim is added. -/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
open VeriTile.Triton

def source_normalization_store_kernel
    (Acc SumExp Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := (triton {
  tl.region Acc = tl.float32, SumExp = tl.float32, Out = tl.float32
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d),
      mask=mask, other=0.0)
  sum_exp = tl.load(SumExp + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), acc / sum_exp, mask=mask)
})

def source_accumulator_step_kernel
    (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut : RegionName)
    (block_seq_n
      head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h
      BLOCK_DMODEL : Nat) :
    ComputeKernel := (triton {
  tl.region Mid_O = tl.float32, Mid_O_LogExpSum = tl.float32, AccIn = tl.float32, MaxLogic = tl.float32, NewMaxLogic = tl.float32, AccOut = tl.float32
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  tv = tl.load(Mid_O + cur_batch * $(stride_mid_ob) +
      cur_head * $(stride_mid_oh) + $(block_seq_n) * $(stride_mid_os) +
      offs_d * $(stride_mid_od), mask=mask, other=0.0)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  acc = tl.load(AccIn + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d),
      mask=mask, other=0.0)
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.load(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  old_scale = tl.exp(max_logic - new_max_logic)
  acc = acc * old_scale
  exp_logic = tl.exp(tlogic - new_max_logic)
  acc = acc + exp_logic * tv
  tl.store(AccOut + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d), acc, mask=mask)
})

def source_sum_exp_step_kernel
    (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h
      stride_logic_b stride_logic_h : Nat) :
    ComputeKernel := (triton {
  tl.region Mid_O_LogExpSum = tl.float32, SumExpIn = tl.float32, MaxLogic = tl.float32, NewMaxLogic = tl.float32, SumExpOut = tl.float32
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  sum_exp = tl.load(SumExpIn + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.load(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  old_scale = tl.exp(max_logic - new_max_logic)
  exp_logic = tl.exp(tlogic - new_max_logic)
  sum_exp = sum_exp * old_scale + exp_logic
  tl.store(SumExpOut + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h), sum_exp)
})

def source_final_store_slice
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := (triton {
  tl.region Final = tl.float32, Out = tl.float32
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d),
      mask=mask, other=0.0)
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final, mask=mask)
})

end VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
