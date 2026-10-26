import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.CallSignature

/-! Complete selected Llama/Phi function bodies and their existing shared-base
candidates. Runtime stride bindings, natural INT32 metadata storage and constexpr
blocks are supplied outside each quotation by the canonical wrapper ABI.
Every source PID/length INT64 conversion, floating operation and mask remains. -/
namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
noncomputable section

def llamaCallSignature : CallSignature where
  pointers := [("B_Seqlen",.i32),("Mid_O",.fp32),("Mid_O_LogExpSum",.fp32),("O",.fp32)]
  scalars := [("stride_mid_ob",.integer),("stride_mid_oh",.integer),("stride_mid_os",.integer),("stride_mid_od",.integer),("stride_mid_o_eb",.integer),("stride_mid_o_eh",.integer),("stride_mid_o_es",.integer),("stride_obs",.integer),("stride_oh",.integer),("stride_od",.integer)]
  constexpr := ["BLOCK_SEQ","BLOCK_DMODEL"]
  readPorts := ["B_Seqlen","Mid_O","Mid_O_LogExpSum"]
  writePorts := ["O"]
  naturalIntegerPointers := ["B_Seqlen"]

def llama (H S D Q BD : Nat) : ComputeKernel :=
  let B_Seqlen : Region .nat := ⟨"L"⟩
  let Mid_O : RegionName := "M"
  let Mid_O_LogExpSum : RegionName := "E"
  let O : RegionName := "O"
  let BLOCK_SEQ := Q
  let BLOCK_DMODEL := BD
  with_triton_call llamaCallSignature call_args [
    stride_mid_ob := H * S * D,
    stride_mid_oh := S * D,
    stride_mid_os := D,
    stride_mid_od := 1,
    stride_mid_o_eb := H * S,
    stride_mid_o_eh := S,
    stride_mid_o_es := 1,
    stride_obs := H * D,
    stride_oh := D,
    stride_od := 1] in triton {
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

def llamaSharedBase (H S D Q BD : Nat) : ComputeKernel :=
  let B_Seqlen : Region .nat := ⟨"L"⟩
  let Mid_O : RegionName := "M"
  let Mid_O_LogExpSum : RegionName := "E"
  let O : RegionName := "O"
  let BLOCK_SEQ := Q
  let BLOCK_DMODEL := BD
  with_triton_call llamaCallSignature call_args [
    stride_mid_ob := H * S * D,
    stride_mid_oh := S * D,
    stride_mid_os := D,
    stride_mid_od := 1,
    stride_mid_o_eb := H * S,
    stride_mid_o_eh := S,
    stride_mid_o_es := 1,
    stride_obs := H * D,
    stride_oh := D,
    stride_od := 1] in triton {
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

def phiCallSignature : CallSignature where
  pointers := [("B_Seqlen",.i32),("Mid_O",.fp32),("Mid_O_LogExpSum",.fp32),("Out",.fp32)]
  scalars := [("stride_mid_ob",.integer),("stride_mid_oh",.integer),("stride_mid_os",.integer),("stride_mid_od",.integer),("stride_mid_o_eb",.integer),("stride_mid_o_eh",.integer),("stride_mid_o_es",.integer),("stride_obs",.integer),("stride_oh",.integer),("stride_od",.integer),("head_dim",.integer)]
  constexpr := ["BLOCK_SEQ","BLOCK_DMODEL"]
  readPorts := ["B_Seqlen","Mid_O","Mid_O_LogExpSum"]
  writePorts := ["Out"]
  naturalIntegerPointers := ["B_Seqlen"]

def phi (H S D Q BD : Nat) : ComputeKernel :=
  let B_Seqlen : Region .nat := ⟨"L"⟩
  let Mid_O : RegionName := "M"
  let Mid_O_LogExpSum : RegionName := "E"
  let Out : RegionName := "O"
  let BLOCK_SEQ := Q
  let BLOCK_DMODEL := BD
  with_triton_call phiCallSignature call_args [
    stride_mid_ob := H * S * D,
    stride_mid_oh := S * D,
    stride_mid_os := D,
    stride_mid_od := 1,
    stride_mid_o_eb := H * S,
    stride_mid_o_eh := S,
    stride_mid_o_es := 1,
    stride_obs := H * D,
    stride_oh := D,
    stride_od := 1,
    head_dim := D] in triton {
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
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os, mask=offs_d < head_dim, other=0.0)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp, mask=offs_d < head_dim)
    return
  }

def phiSharedBase (H S D Q BD : Nat) : ComputeKernel :=
  let B_Seqlen : Region .nat := ⟨"L"⟩
  let Mid_O : RegionName := "M"
  let Mid_O_LogExpSum : RegionName := "E"
  let Out : RegionName := "O"
  let BLOCK_SEQ := Q
  let BLOCK_DMODEL := BD
  with_triton_call phiCallSignature call_args [
    stride_mid_ob := H * S * D,
    stride_mid_oh := S * D,
    stride_mid_os := D,
    stride_mid_od := 1,
    stride_mid_o_eb := H * S,
    stride_mid_o_eh := S,
    stride_mid_o_es := 1,
    stride_obs := H * D,
    stride_oh := D,
    stride_od := 1,
    head_dim := D] in triton {
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
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os, mask=offs_d < head_dim, other=0.0)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp, mask=offs_d < head_dim)
    return
  }

def kernel (masked : Bool) (H S D Q BD : Nat) : ComputeKernel :=
  if masked then phi H S D Q BD else llama H S D Q BD

end
end VeriTile.Bench.Repairs.FlashDecode2
