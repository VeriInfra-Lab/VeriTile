import VeriTile.Triton.DSL.CallSignature

/-! Complete original and pre-existing address-first Python bodies. Pointer
storage and per-call scalar values belong to the caller. The selected wrapper
uses FP32 values and signed INT32 metadata. Nonnegative start/length carriers
retain the existing IO refinement; native width bounds remain in the contract.
-/
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def signature : CallSignature where
  pointers := [("Logics",.fp32),("V",.fp32),("Out",.fp32),
    ("B_Loc",.i32),("B_Start_Loc",.i32),("B_Seqlen",.i32)]
  naturalIntegerPointers := ["B_Start_Loc","B_Seqlen"]
  typedFP32Storage := Bool.true
  scalars := [("max_input_len",.integer),("stride_logic_h",.integer),("stride_logic_bs",.integer),("stride_vbs",.integer),("stride_vh",.integer),("stride_vd",.integer),("stride_obs",.integer),("stride_oh",.integer),("stride_od",.integer),("stride_b_loc_b",.integer),("stride_b_loc_s",.integer),("other_kv_index",.integer)]
  constexpr := ["BLOCK_DMODEL","BLOCK_N"]

/-- Complete `_fwd_kernel` from the selected Python file. -/
def sourceBody (Logics V Out : RegionName) (B_Loc : Region .int)
    (B_Start_Loc B_Seqlen : Region .nat)
    (max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat) (other_kv_index : Int) : ComputeKernel :=
  with_triton_call signature call_args [max_input_len := max_input_len, stride_logic_h := stride_logic_h, stride_logic_bs := stride_logic_bs, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, stride_b_loc_b := stride_b_loc_b, stride_b_loc_s := stride_b_loc_s, other_kv_index := other_kv_index] in
  triton {
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs,
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs,
                    mask=(start_n + offs_n < cur_batch_seq_len)[:, None], other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return
  }

/-- Complete `_fwd_kernel_early_address` from the selected Python file. -/
def earlyAddressBody (Logics V Out : RegionName) (B_Loc : Region .int)
    (B_Start_Loc B_Seqlen : Region .nat)
    (max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat) (other_kv_index : Int) : ComputeKernel :=
  with_triton_call signature call_args [max_input_len := max_input_len, stride_logic_h := stride_logic_h, stride_logic_bs := stride_logic_bs, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, stride_b_loc_b := stride_b_loc_b, stride_b_loc_s := stride_b_loc_s, other_kv_index := other_kv_index] in
  triton {
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs,
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs,
                    mask=(start_n + offs_n < cur_batch_seq_len)[:, None], other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    acc = acc / e_sum
    tl.store(out_ptrs, acc)
    return
  }

structure Parameters where
  (logics values output : RegionName)
  locations : Region .int
  (starts lengths : Region .nat)
  (maxLength logicHeadStride logicSequenceStride valueBatchStride valueHeadStride valueDimStride
    outputBatchStride outputHeadStride outputDimStride locationBatchStride locationSequenceStride
    dimensions tokens : Nat)
  sentinel : Int

def source (p : Parameters) : ComputeKernel :=
  sourceBody p.logics p.values p.output p.locations p.starts p.lengths p.maxLength
    p.logicHeadStride p.logicSequenceStride p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.locationBatchStride p.locationSequenceStride
    p.dimensions p.tokens p.sentinel

end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.EarlyAddress
open VeriTile.Triton

def source (p : FPValue.Parameters) : ComputeKernel :=
  FPValue.earlyAddressBody p.logics p.values p.output p.locations p.starts p.lengths p.maxLength
    p.logicHeadStride p.logicSequenceStride p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.locationBatchStride p.locationSequenceStride
    p.dimensions p.tokens p.sentinel

end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.EarlyAddress
