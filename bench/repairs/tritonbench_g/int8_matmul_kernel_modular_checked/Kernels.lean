import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature

/-! Complete unchanged selected Python function body. Storage, runtime call
bindings and constexpr values are outside; the original add.u32 instruction
and its complete inline-assembly ABI remain visible in the quotation. -/
namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

def callSignature : CallSignature where
  pointers := [("a_ptr",.fp32),("b_ptr",.u8),("c_ptr",.i32)]
  scalars := [("M",.integer),("N",.integer),("stride_am",.integer),
    ("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),
    ("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["K","BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def kernel (a_ptr : Region .fp32) (b_ptr : Region .nat) (c_ptr : Region .int)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [M := M, N := N,
    stride_am := stride_am, stride_ak := stride_ak,
    stride_bk := stride_bk, stride_bn := stride_bn,
    stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
  tl.static_assert(
      K % (4 * BLOCK_SIZE_K) == 0,
      "K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K",
  )
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  num_pid_in_group = GROUP_SIZE_M * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * GROUP_SIZE_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
  offs_k = tl.arange(0, BLOCK_SIZE_K)
  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
  for i in range(4):
      b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
      for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K)):
          k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j
          a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0).to(tl.int8)  # Cast to int8.
          b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
          mask = 3 << (2 * i)
          b = ((b_uint8 & mask) >> (2 * i)).to(tl.int8)  # Cast to int8.
          tensor_full = tl.full((1,), 1, dtype=tl.int8)  # Use int8.
          # The zero-seeded block sum fits INT32 for all retained BK <= 128.
          partial = tl.dot(a, (b - tensor_full), out_dtype=tl.int32)
          accumulator = tl.inline_asm_elementwise(
              "add.u32 $0, $1, $2;",
              constraints="=r,r,r",
              args=[accumulator, partial],
              dtype=tl.int32,
              is_pure=True,
              pack=1,
          )
          a_ptrs += BLOCK_SIZE_K * stride_ak
          b_ptrs += BLOCK_SIZE_K * stride_bk
  c = accumulator
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)
  }

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
