import bench.tritonbench_g.matmul_dequant_int4.Kernels
import VeriTile.Triton.DSL.PackedIntegerPromotion
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.Body
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 6000000

def original_matmul4_kernel_signature (dtype : FloatDType) : CallSignature where
  pointers := [("b_ptr",.i32), ("zeros_ptr",.i32)]
  floatingPointers := [("a_ptr",dtype), ("c_ptr",dtype), ("scales_ptr",dtype)]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer), ("stride_scales_g",.integer), ("stride_scales_n",.integer), ("stride_zeros_g",.integer), ("stride_zeros_n",.integer), ("groupsize",.integer)]
  constexpr := ["NO_GROUPS", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"]

def original_matmul4_kernel (dtype : FloatDType) (a_ptr b_ptr c_ptr scales_ptr zeros_ptr : RegionName) (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n groupsize BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) (NO_GROUPS : Bool) : ComputeKernel :=
  with_triton_call (original_matmul4_kernel_signature dtype) call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn, stride_scales_g := stride_scales_g, stride_scales_n := stride_scales_n, stride_zeros_g := stride_zeros_g, stride_zeros_n := stride_zeros_n, groupsize := groupsize] in triton {
      """
      Compute the matrix multiplication C = A x B.
      A is of shape (M, K) float16
      B is of shape (K//8, N) int32
      C is of shape (M, N) float16
      scales is of shape (G, N) float16
      zeros is of shape (G, N//8) int32
      groupsize is an int specifying the size of groups for scales and zeros.
      G is K // groupsize.
      Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
      WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
      WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
      WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
      """
      bits = 4
      infearure_per_bits = 8
      pid = tl.program_id(axis=0)
      num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
      num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
      num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
      num_pid_in_group = GROUP_SIZE_M * num_pid_n
      group_id = pid // num_pid_in_group
      first_pid_m = group_id * GROUP_SIZE_M
      group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
      pid_m = first_pid_m + (pid % group_size_m)
      pid_n = (pid % num_pid_in_group) // group_size_m
      offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      offs_k = tl.arange(0, BLOCK_SIZE_K)
      a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
      a_mask = (offs_am[:, None] < M)
      b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
      scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
      zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
      shifter = (offs_k % infearure_per_bits) * bits
      zeros_shifter = (offs_bn % infearure_per_bits) * bits
      if NO_GROUPS:
          scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
          zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32
          zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
          zeros = zeros * scales
      accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      for k in range(0, num_pid_k):
          a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
          b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
          if not NO_GROUPS:
              g_id = k // (groupsize // BLOCK_SIZE_K)
              ptr = scales_ptrs + g_id * stride_scales_g
              scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
              ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
              zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32
              zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
              zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16
          b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
          b = b * scales[None, :] - zeros[None, :]  # Scale and shift
          accumulator += tl.dot(a, b)
          a_ptrs += BLOCK_SIZE_K * stride_ak
          b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
      c = accumulator.to(tl.float16)
      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
      tl.store(c_ptrs, accumulator, mask=c_mask)
  }

def original_matmul_kernel_signature (dtype : FloatDType) : CallSignature where
  pointers := [("b_ptr",.i32), ("bzp_ptr",.i32), ("c_ptr",.fp16)]
  floatingPointers := [("a_ptr",dtype), ("bs_ptr",dtype)]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer), ("stride_bsk",.integer), ("stride_bsn",.integer), ("stride_bzpk",.integer), ("stride_bzpn",.integer), ("group_size",.integer)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "SPLIT_K"]

def original_matmul_kernel (dtype : FloatDType) (a_ptr b_ptr c_ptr bs_ptr bzp_ptr : RegionName) (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_bsk stride_bsn stride_bzpk stride_bzpn group_size BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call (original_matmul_kernel_signature dtype) call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn, stride_bsk := stride_bsk, stride_bsn := stride_bsn, stride_bzpk := stride_bzpk, stride_bzpn := stride_bzpn, group_size := group_size] in triton {
      """
      Matrix multiplication kernel with dequantization for 4-bit integers
      A: [M, K], B: [K//8, N], C: [M, N]
      """
      pid = tl.program_id(axis=0)
      pid_sp_k = tl.program_id(axis=1)
      num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
      num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
      num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
      num_pid_in_group = GROUP_SIZE_M * num_pid_n
      group_id = pid // num_pid_in_group
      first_pid_m = group_id * GROUP_SIZE_M
      group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
      pid_m = first_pid_m + (pid % group_size_m)
      pid_n = (pid % num_pid_in_group) // group_size_m
      offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
      offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
      offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

      a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
      b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

      accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
          bs_ptrs = bs_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
              + offs_bn[None, :] * stride_bsn
          bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
              + (offs_bn[None, :] // 8) * stride_bzpn
          b_shift_bits = (offs_k[:, None] % 8) * 4
          bzp_shift_bits = (offs_bn[None, :] % 8) * 4
          a = tl.load(a_ptrs, mask=(offs_k[None, :] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
          b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
          bs = tl.load(bs_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
          bzp = tl.load(bzp_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)

          int_b = (b >> b_shift_bits) & 0xF
          int_bzp = (bzp >> bzp_shift_bits) & 0xF
          b = ((int_b - int_bzp) * bs).to(tl.float16)
          accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16))
          a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
          b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)

      c = accumulator.to(tl.float16)
      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
      if SPLIT_K == 1:
          tl.store(c_ptrs, c, mask=c_mask)
      else:
          tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def candidate_matmul4_kernel_signature (dtype : FloatDType) : CallSignature where
  pointers := [("b_ptr",.i32), ("zeros_ptr",.i32)]
  floatingPointers := [("a_ptr",dtype), ("c_ptr",dtype), ("scales_ptr",dtype)]
  scalars := [("M",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer), ("stride_scales_g",.integer), ("stride_scales_n",.integer), ("stride_zeros_g",.integer), ("stride_zeros_n",.integer), ("groupsize",.integer)]
  constexpr := ["N", "NO_GROUPS", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"]

def candidate_matmul4_kernel (dtype : FloatDType) (a_ptr b_ptr c_ptr scales_ptr zeros_ptr : RegionName) (M K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n groupsize N BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) (NO_GROUPS : Bool) : ComputeKernel :=
  with_triton_call (candidate_matmul4_kernel_signature dtype) call_args [M := M, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn, stride_scales_g := stride_scales_g, stride_scales_n := stride_scales_n, stride_zeros_g := stride_zeros_g, stride_zeros_n := stride_zeros_n, groupsize := groupsize] in triton {
      """
      Compute the matrix multiplication C = A x B.
      A is of shape (M, K) float16
      B is of shape (K//8, N) int32
      C is of shape (M, N) float16
      scales is of shape (G, N) float16
      zeros is of shape (G, N//8) int32
      groupsize is an int specifying the size of groups for scales and zeros.
      G is K // groupsize.
      Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
      WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
      WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
      WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
      """
      bits = 4
      infearure_per_bits = 8
      pid = tl.program_id(axis=0)
      num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
      num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
      num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
      num_pid_in_group: tl.constexpr = GROUP_SIZE_M * triton.cdiv(N, BLOCK_SIZE_N)
      group_id = pid // num_pid_in_group
      first_pid_m = group_id * GROUP_SIZE_M
      group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
      pid_m = first_pid_m + (pid % group_size_m)
      pid_n = (pid % num_pid_in_group) // group_size_m
      offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      offs_k = tl.arange(0, BLOCK_SIZE_K)
      a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
      a_mask = (offs_am[:, None] < M)
      b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
      scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
      zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
      shifter = (offs_k % infearure_per_bits) * bits
      zeros_shifter = (offs_bn % infearure_per_bits) * bits
      if NO_GROUPS:
          scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
          zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32
          zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
          zeros = zeros * scales
      accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      for k in range(0, num_pid_k):
          a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
          b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
          if not NO_GROUPS:
              g_id = k // (groupsize // BLOCK_SIZE_K)
              ptr = scales_ptrs + g_id * stride_scales_g
              scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
              ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
              zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32
              zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
              zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16
          b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
          b = b * scales[None, :] - zeros[None, :]  # Scale and shift
          accumulator += tl.dot(a, b)
          a_ptrs += BLOCK_SIZE_K * stride_ak
          b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
      c = accumulator.to(tl.float16)
      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
      tl.store(c_ptrs, accumulator, mask=c_mask)
  }

def candidate_matmul_kernel_signature (dtype : FloatDType) : CallSignature where
  pointers := [("b_ptr",.i32), ("bzp_ptr",.i32), ("c_ptr",.fp16)]
  floatingPointers := [("a_ptr",dtype), ("bs_ptr",dtype)]
  scalars := [("M",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer), ("stride_bsk",.integer), ("stride_bsn",.integer), ("stride_bzpk",.integer), ("stride_bzpn",.integer), ("group_size",.integer)]
  constexpr := ["N", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "SPLIT_K"]

def candidate_matmul_kernel (dtype : FloatDType) (a_ptr b_ptr c_ptr bs_ptr bzp_ptr : RegionName) (M K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_bsk stride_bsn stride_bzpk stride_bzpn group_size N BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M SPLIT_K : Nat) : ComputeKernel :=
  with_triton_call (candidate_matmul_kernel_signature dtype) call_args [M := M, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn, stride_bsk := stride_bsk, stride_bsn := stride_bsn, stride_bzpk := stride_bzpk, stride_bzpn := stride_bzpn, group_size := group_size] in triton {
      """
      Matrix multiplication kernel with dequantization for 4-bit integers
      A: [M, K], B: [K//8, N], C: [M, N]
      """
      pid = tl.program_id(axis=0)
      pid_sp_k = tl.program_id(axis=1)
      num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
      num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
      num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
      num_pid_in_group: tl.constexpr = GROUP_SIZE_M * triton.cdiv(N, BLOCK_SIZE_N)
      group_id = pid // num_pid_in_group
      first_pid_m = group_id * GROUP_SIZE_M
      group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
      pid_m = first_pid_m + (pid % group_size_m)
      pid_n = (pid % num_pid_in_group) // group_size_m
      offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
      offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
      offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

      a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
      b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

      accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
          bs_ptrs = bs_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
              + offs_bn[None, :] * stride_bsn
          bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
              + (offs_bn[None, :] // 8) * stride_bzpn
          b_shift_bits = (offs_k[:, None] % 8) * 4
          bzp_shift_bits = (offs_bn[None, :] % 8) * 4
          a = tl.load(a_ptrs, mask=(offs_k[None, :] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
          b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
          bs = tl.load(bs_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
          bzp = tl.load(bzp_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)

          int_b = (b >> b_shift_bits) & 0xF
          int_bzp = (bzp >> bzp_shift_bits) & 0xF
          b = ((int_b - int_bzp) * bs).to(tl.float16)
          accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16))
          a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
          b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)

      c = accumulator.to(tl.float16)
      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
      if SPLIT_K == 1:
          tl.store(c_ptrs, c, mask=c_mask)
      else:
          tl.atomic_add(c_ptrs, c, mask=c_mask)
  }

def candidate_dequantize_kernel_signature (dtype : FloatDType) : CallSignature where
  pointers := [("b_ptr",.i32), ("b_zp_ptr",.i32)]
  floatingPointers := [("b_scale_ptr",dtype), ("fpb_ptr",dtype)]
  scalars := [("K",.integer), ("N",.integer), ("group_size",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_bsk",.integer), ("stride_bsn",.integer), ("stride_bzpk",.integer), ("stride_bzpn",.integer), ("stride_fpbk",.integer), ("stride_fpbn",.integer)]
  constexpr := ["BLOCK_SIZE_K", "BLOCK_SIZE_N"]

def candidate_dequantize_kernel (dtype : FloatDType) (b_ptr b_scale_ptr b_zp_ptr fpb_ptr : RegionName) (K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn stride_fpbk stride_fpbn BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call (candidate_dequantize_kernel_signature dtype) call_args [K := K, N := N, group_size := group_size, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsk := stride_bsk, stride_bsn := stride_bsn, stride_bzpk := stride_bzpk, stride_bzpn := stride_bzpn, stride_fpbk := stride_fpbk, stride_fpbn := stride_fpbn] in triton {
      """Dequantize tile [BLOCK_SIZE_K, BLOCK_SIZE_N] in full precision.
      We should assert BLOCK_SIZE_N % 8 == 0.
      weight[K // 8, N], scale[K // group_size, N], zp[K // group_size, N // group_size]
      """
      k_block_idx = tl.program_id(axis=0)
      n_block_idx = tl.program_id(axis=1)
      offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
      offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      fpb_offs = k_block_idx * (BLOCK_SIZE_K * stride_fpbk) + n_block_idx * (BLOCK_SIZE_N * stride_fpbn) + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_fpbk + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_fpbn
      b_offs = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
      bzp_offs = (offs_k[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
      bs_offs = (offs_k[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
      n_mask = offs_n[None, :] < N
      k_mask = offs_k[:, None] < K
      mask = n_mask & k_mask
      int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
      zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
      scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
      b_shift = (offs_k[:, None] % 8) * 4
      bzp_shift = (offs_n[None, :] % 8) * 4
      fp_weight = (((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b
      tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
  }

end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.Body
