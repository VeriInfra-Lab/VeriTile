import bench.tritonbench_g.iv_dependent_matmul.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.StringCallControl

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
open Lean VeriTile.Triton
open TritonBenchG.IvDependentMatmul FPValue
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 8000000

def callSignature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("a_ptr",dtype),("b_ptr",dtype),("c_ptr",dtype)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),
    ("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),
    ("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","type"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def originalKernel (dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (type : String) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [M := M, N := N, K := K,
    stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk,
    stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
  M = tl.cast(M, tl.int64)
  N = tl.cast(N, tl.int64)
  K = tl.cast(K, tl.int64)
  stride_am = tl.cast(stride_am, tl.int64)
  stride_ak = tl.cast(stride_ak, tl.int64)
  stride_bk = tl.cast(stride_bk, tl.int64)
  stride_bn = tl.cast(stride_bn, tl.int64)
  stride_cm = tl.cast(stride_cm, tl.int64)
  stride_cn = tl.cast(stride_cn, tl.int64)
  pid = tl.cast(tl.program_id(axis=0), tl.int64)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n

  offs_am = (pid_m * BLOCK_SIZE_M + tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.cast(tl.arange(0, BLOCK_SIZE_N), tl.int64)) % N
  offs_k = tl.cast(tl.arange(0, BLOCK_SIZE_K), tl.int64)
  a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  if type == "post_load_two_iters":
      a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
      b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
  elif type == "post_load_three_iters":
      a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
      b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
      a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
      b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
      k64 = tl.cast(k, tl.int64)
      if type == "pre_load":
          a_ptrs = a_ptr + k64 * BLOCK_SIZE_K * stride_ak
          b_ptrs = b_ptr + k64 * BLOCK_SIZE_K * stride_bk
      elif type == "post_pre_mixed":
          a_ptrs = a_ptr + k64 * BLOCK_SIZE_K * stride_ak
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k64 * BLOCK_SIZE_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k64 * BLOCK_SIZE_K, other=0.0)
      accumulator += tl.dot(a, b)
      if type == "post_load":
          a_ptrs = a_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_ak
          b_ptrs = b_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_bk
      elif type == "post_pre_mixed":
          b_ptrs = b_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_bk
      elif type == "post_load_two_iters":
          a_ptrs = a_ptrs_next
          b_ptrs = b_ptrs_next
          a_ptrs_next = a_ptr + (k64 + 2) * BLOCK_SIZE_K * stride_ak
          b_ptrs_next = b_ptr + (k64 + 2) * BLOCK_SIZE_K * stride_bk
      elif type == "post_load_three_iters":
          a_ptrs = a_ptrs_next
          b_ptrs = b_ptrs_next
          a_ptrs_next = a_ptrs_next_next
          b_ptrs_next = b_ptrs_next_next
          a_ptrs_next_next = a_ptr + (k64 + 3) * BLOCK_SIZE_K * stride_ak
          b_ptrs_next_next = b_ptr + (k64 + 3) * BLOCK_SIZE_K * stride_bk
  c = accumulator.to(tl.float16)

  offs_cm = pid_m * BLOCK_SIZE_M + tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.cast(tl.arange(0, BLOCK_SIZE_N), tl.int64)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)
  }

def nativeKernel (dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (type : String) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [M := M, N := N, K := K,
    stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk,
    stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
  M = tl.cast(M, tl.int64)
  N = tl.cast(N, tl.int64)
  K = tl.cast(K, tl.int64)
  stride_am = tl.cast(stride_am, tl.int64)
  stride_ak = tl.cast(stride_ak, tl.int64)
  stride_bk = tl.cast(stride_bk, tl.int64)
  stride_bn = tl.cast(stride_bn, tl.int64)
  stride_cm = tl.cast(stride_cm, tl.int64)
  stride_cn = tl.cast(stride_cn, tl.int64)
  pid = tl.cast(tl.program_id(axis=0), tl.int64)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n

  offs_am = (pid_m * BLOCK_SIZE_M + tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.cast(tl.arange(0, BLOCK_SIZE_N), tl.int64)) % N
  offs_k = tl.cast(tl.arange(0, BLOCK_SIZE_K), tl.int64)
  a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  if type == "post_load_two_iters":
      a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
      b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
  elif type == "post_load_three_iters":
      a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
      b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
      a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
      b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
      k64 = tl.cast(k, tl.int64)
      if type == "pre_load":
          a_ptrs = a_ptr + k64 * BLOCK_SIZE_K * stride_ak
          b_ptrs = b_ptr + k64 * BLOCK_SIZE_K * stride_bk
      elif type == "post_pre_mixed":
          a_ptrs = a_ptr + k64 * BLOCK_SIZE_K * stride_ak
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k64 * BLOCK_SIZE_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k64 * BLOCK_SIZE_K, other=0.0)
      accumulator = tl.dot(a, b, accumulator)
      if type == "post_load":
          a_ptrs = a_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_ak
          b_ptrs = b_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_bk
      elif type == "post_pre_mixed":
          b_ptrs = b_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_bk
      elif type == "post_load_two_iters":
          a_ptrs = a_ptrs_next
          b_ptrs = b_ptrs_next
          a_ptrs_next = a_ptr + (k64 + 2) * BLOCK_SIZE_K * stride_ak
          b_ptrs_next = b_ptr + (k64 + 2) * BLOCK_SIZE_K * stride_bk
      elif type == "post_load_three_iters":
          a_ptrs = a_ptrs_next
          b_ptrs = b_ptrs_next
          a_ptrs_next = a_ptrs_next_next
          b_ptrs_next = b_ptrs_next_next
          a_ptrs_next_next = a_ptr + (k64 + 3) * BLOCK_SIZE_K * stride_ak
          b_ptrs_next_next = b_ptr + (k64 + 3) * BLOCK_SIZE_K * stride_bk
  c = accumulator.to(tl.float16)

  offs_cm = pid_m * BLOCK_SIZE_M + tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.cast(tl.arange(0, BLOCK_SIZE_N), tl.int64)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)
  }

def preLoadSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  originalKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "pre_load"

def preLoadNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  nativeKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "pre_load"

def postLoadSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  originalKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_load"

def postLoadNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  nativeKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_load"

def mixedSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  originalKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_pre_mixed"

def mixedNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  nativeKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_pre_mixed"

def twoSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  originalKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_load_two_iters"

def twoNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  nativeKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_load_two_iters"

def threeSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  originalKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_load_three_iters"

def threeNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  nativeKernel dtype A B C M N K sam sak sbk sbn scm scn BM BN BK "post_load_three_iters"

def quotation (s : Kernels.Schedule) (g : Graph) (p : Parameters) : ComputeKernel :=
  (match g with | .source => originalKernel | .accumulator => nativeKernel)
    .fp32 p.input p.weight p.output p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
    p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
    p.blockRows p.blockColumns p.blockInner s.name

end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
