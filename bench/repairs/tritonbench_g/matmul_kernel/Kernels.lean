import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.matmul_kernel.HostModel

namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton

/-- Source pointer dtypes and call values; native signed32 bounds remain in
its existing public input contract, independently of the integer carrier. -/
def callSignature : CallSignature where
  pointers := [("c_ptr",.fp32),("a_ptr",.fp32),("b_ptr",.fp32)]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]
  readPorts := ["a_ptr", "b_ptr"]
  writePorts := ["c_ptr"]

/-- Unchanged function body from `matmul_kernel_repaired.py:matmul_kernel`. -/
def typed (C A B : RegionName) (cfg : Config) : ComputeKernel :=
  let c_ptr : VeriTile.Triton.Region .fp32 := C.cast
  let a_ptr : VeriTile.Triton.Region .fp32 := A.cast
  let b_ptr : VeriTile.Triton.Region .fp32 := B.cast
  let BLOCK_SIZE_M := cfg.BM
  let BLOCK_SIZE_N := cfg.BN
  let BLOCK_SIZE_K := cfg.BK
  with_triton_call callSignature call_args [M := cfg.M, N := cfg.N, K := cfg.K, stride_am := cfg.A.sr, stride_ak := cfg.A.sc, stride_bk := cfg.B.sr, stride_bn := cfg.B.sc, stride_cm := cfg.C.sr, stride_cn := cfg.C.sc] in
  triton {
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
  }

/-- Unchanged function body from `matmul_kernel_remaining_k.py:matmul_kernel`. -/
def typedRemaining (C A B : RegionName) (cfg : Config) : ComputeKernel :=
  let c_ptr : VeriTile.Triton.Region .fp32 := C.cast
  let a_ptr : VeriTile.Triton.Region .fp32 := A.cast
  let b_ptr : VeriTile.Triton.Region .fp32 := B.cast
  let BLOCK_SIZE_M := cfg.BM
  let BLOCK_SIZE_N := cfg.BN
  let BLOCK_SIZE_K := cfg.BK
  with_triton_call callSignature call_args [M := cfg.M, N := cfg.N, K := cfg.K, stride_am := cfg.A.sr, stride_ak := cfg.A.sc, stride_bk := cfg.B.sr, stride_bn := cfg.B.sc, stride_cm := cfg.C.sr, stride_cn := cfg.C.sc] in
  triton {
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
  }

def source (cfg : Config) := typed "C" "A" "B" cfg
def remaining (cfg : Config) := typedRemaining "C" "A" "B" cfg

end VeriTile.Bench.Repairs.MatmulKernel.KernelSource
