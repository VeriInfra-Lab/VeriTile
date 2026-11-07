import VeriTile.Triton.DSL.CallSignature

/-! The complete original Python body and existing metadata-address candidate.
Int32 metadata, int64 widening and every FP32 conversion remain explicit.
Bloom shares these bodies because its original and candidate are identical. -/
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open VeriTile.Triton
def callSignature (dtype : FloatDType) : CallSignature where
  pointers := [("B_Start_Loc",.i32),("B_Seqlen",.i32)]
  floatingPointers := [("Logics",dtype),("Prob_Out",dtype)]
  scalars := [("stride_start",.i64),("stride_length",.i64),("stride_logic_h",.i64),
    ("stride_logic_bs",.i64),("stride_prob_h",.i64),("stride_prob_bs",.i64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["B_Seqlen","B_Start_Loc","Logics"]
  writePorts := ["Prob_Out"]

def sourceKernel (dtype : FloatDType)
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .int) (Prob_Out : RegionName)
    (stride_start stride_length stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel :=
  with_triton_call (callSignature dtype) call_args
    [stride_start := stride_start, stride_length := stride_length, stride_logic_h := stride_logic_h,
      stride_logic_bs := stride_logic_bs, stride_prob_h := stride_prob_h, stride_prob_bs := stride_prob_bs] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_length).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch * stride_start).to(tl.int64)
    row = tl.load(Logics + cur_head * stride_logic_h +
                  (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs,
                  mask=col_offsets < cur_batch_seq_len, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(Prob_Out + cur_head * stride_prob_h +
             (cur_batch_in_all_start_index + col_offsets) * stride_prob_bs,
             softmax_output, mask=col_offsets < cur_batch_seq_len)
  }

structure Parameters where
  (input output : RegionName)
  (start length : Region .int)
  (startStride lengthStride : Nat)
  (inputHeadStride inputTokenStride outputHeadStride outputTokenStride width : Nat)

def kernel (d : FloatDType) (p : Parameters) := sourceKernel d p.input p.start p.length p.output
  p.startStride p.lengthStride p.inputHeadStride p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width

end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelSource
open VeriTile.Triton Strided

def indexCommutedKernel (dtype : FloatDType)
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .int) (Prob_Out : RegionName)
    (stride_start stride_length stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel :=
  with_triton_call (callSignature dtype) call_args
    [stride_start := stride_start, stride_length := stride_length, stride_logic_h := stride_logic_h,
      stride_logic_bs := stride_logic_bs, stride_prob_h := stride_prob_h, stride_prob_bs := stride_prob_bs] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + stride_length * cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch * stride_start).to(tl.int64)
    row = tl.load(Logics + cur_head * stride_logic_h +
                  (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs,
                  mask=col_offsets < cur_batch_seq_len, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(Prob_Out + cur_head * stride_prob_h +
             (cur_batch_in_all_start_index + col_offsets) * stride_prob_bs,
             softmax_output, mask=col_offsets < cur_batch_seq_len)
  }

def candidateKernel (p : Parameters) := indexCommutedKernel .fp32 p.input p.start p.length p.output
  p.startStride p.lengthStride p.inputHeadStride p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelSource
