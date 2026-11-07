import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

/-! Literal original and retained shared-base candidate bodies. Pointer storage,
all runtime strides and constexpr dimensions are supplied outside the bodies. -/
namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton
def llamaCallSignature : CallSignature where
  pointers := [("B_Seqlen",.i32),("Mid_O",.fp32),("Mid_O_LogExpSum",.fp32),("O",.fp32)]
  scalars := [("stride_mid_ob",.integer), ("stride_mid_oh",.integer), ("stride_mid_os",.integer), ("stride_mid_od",.integer), ("stride_mid_o_eb",.integer), ("stride_mid_o_eh",.integer), ("stride_mid_o_es",.integer), ("stride_obs",.integer), ("stride_oh",.integer), ("stride_od",.integer)]
  constexpr := ["BLOCK_SEQ","BLOCK_DMODEL"]
  readPorts := ["B_Seqlen","Mid_O","Mid_O_LogExpSum"]
  writePorts := ["O"]

def llamaKernel (H S D Q BD : Nat) : ComputeKernel :=
  let B_Seqlen : Region .int := ⟨"L"⟩
  let Mid_O : RegionName := "M"
  let Mid_O_LogExpSum : RegionName := "E"
  let O : RegionName := "O"
  let BLOCK_SEQ := Q
  let BLOCK_DMODEL := BD
  let stride_mid_ob : Nat := H*S*D
  let stride_mid_oh : Nat := S*D
  let stride_mid_os : Nat := D
  let stride_mid_od : Nat := 1
  let stride_mid_o_eb : Nat := H*S
  let stride_mid_o_eh : Nat := S
  let stride_mid_o_es : Nat := 1
  let stride_obs : Nat := H*D
  let stride_oh : Nat := D
  let stride_od : Nat := 1
  with_triton_call llamaCallSignature call_args [stride_mid_ob := stride_mid_ob, stride_mid_oh := stride_mid_oh, stride_mid_os := stride_mid_os, stride_mid_od := stride_mid_od, stride_mid_o_eb := stride_mid_o_eb, stride_mid_o_eh := stride_mid_o_eh, stride_mid_o_es := stride_mid_o_es, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return
  }
end VeriTile.Bench.Repairs.FlashDecode2

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.Candidates
open VeriTile.Triton FlashDecode2

def llamaSharedBaseKernel (H S D Q BD : Nat) : ComputeKernel :=
  let B_Seqlen : Region .int := ⟨"L"⟩
  let Mid_O : RegionName := "M"
  let Mid_O_LogExpSum : RegionName := "E"
  let O : RegionName := "O"
  let BLOCK_SEQ := Q
  let BLOCK_DMODEL := BD
  let stride_mid_ob : Nat := H*S*D
  let stride_mid_oh : Nat := S*D
  let stride_mid_os : Nat := D
  let stride_mid_od : Nat := 1
  let stride_mid_o_eb : Nat := H*S
  let stride_mid_o_eh : Nat := S
  let stride_mid_o_es : Nat := 1
  let stride_obs : Nat := H*D
  let stride_oh : Nat := D
  let stride_od : Nat := 1
  with_triton_call llamaCallSignature call_args [stride_mid_ob := stride_mid_ob, stride_mid_oh := stride_mid_oh, stride_mid_os := stride_mid_os, stride_mid_od := stride_mid_od, stride_mid_o_eb := stride_mid_o_eb, stride_mid_o_eh := stride_mid_o_eh, stride_mid_o_es := stride_mid_o_es, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    offs_v = offs_logic * stride_mid_os + offs_d
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return
  }
end VeriTile.Bench.Repairs.FlashDecode2Llama.Candidates
