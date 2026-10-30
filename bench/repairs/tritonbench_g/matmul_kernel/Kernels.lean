import VeriTile.Triton
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.HostKernels

namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton

/-- Complete selected source. Pointer element types come from the admitted
FP32 tensor metadata. The explicit FP32 cast in the store represents Triton's
implicit conversion to C's pointer element type; the source's FP16 narrowing
and live three-argument dot remain in their original positions. -/
def typed (C A B : RegionName) (cfg : Config) : ComputeKernel :=
  let c_ptr : VeriTile.Triton.Region .fp32 := C.cast
  let a_ptr : VeriTile.Triton.Region .fp32 := A.cast
  let b_ptr : VeriTile.Triton.Region .fp32 := B.cast
  triton {
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = (pid_m * $(cfg.BM) + tl.arange(0, $(cfg.BM))) % $(cfg.M)
    offs_bn = (pid_n * $(cfg.BN) + tl.arange(0, $(cfg.BN))) % $(cfg.N)
    offs_k = tl.arange(0, $(cfg.BK))
    a_ptrs = a_ptr + (offs_am[:, None] * $(cfg.A.sr) + offs_k[None, :] * $(cfg.A.sc))
    b_ptrs = b_ptr + (offs_k[:, None] * $(cfg.B.sr) + offs_bn[None, :] * $(cfg.B.sc))
    accumulator = tl.zeros([$(cfg.BM), $(cfg.BN)], dtype=tl.float32)
    for k in range($(0), $(cfg.steps), $(1)) {
      a = tl.load(a_ptrs, mask=offs_k[None, :] + k * $(cfg.BK) < $(cfg.K), other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] + k * $(cfg.BK) < $(cfg.K), other=0.0)
      accumulator = tl.dot(a, b, accumulator)
      a_ptrs += $(cfg.BK) * $(cfg.A.sc)
      b_ptrs += $(cfg.BK) * $(cfg.B.sr)
    }
    c = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * $(cfg.BM) + tl.arange(0, $(cfg.BM))
    offs_cn = pid_n * $(cfg.BN) + tl.arange(0, $(cfg.BN))
    c_ptrs = c_ptr + $(cfg.C.sr) * offs_cm[:, None] + $(cfg.C.sc) * offs_cn[None, :]
    tl.store(c_ptrs, tl.cast(c, tl.float32), mask=(offs_cm[:, None] < $(cfg.M)) & (offs_cn[None, :] < $(cfg.N)))
  }

def source (cfg : Config) := typed "C" "A" "B" cfg

/-- K-tail mask strength reduction: compare each lane with a scalar remaining
length, eliminating the per-lane block-offset addition in both load masks.
All pointers, dot operations, conversion and output statements are retained. -/
def typedRemaining (C A B : RegionName) (cfg : Config) : ComputeKernel :=
  let c_ptr : VeriTile.Triton.Region .fp32 := C.cast
  let a_ptr : VeriTile.Triton.Region .fp32 := A.cast
  let b_ptr : VeriTile.Triton.Region .fp32 := B.cast
  triton {
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = (pid_m * $(cfg.BM) + tl.arange(0, $(cfg.BM))) % $(cfg.M)
    offs_bn = (pid_n * $(cfg.BN) + tl.arange(0, $(cfg.BN))) % $(cfg.N)
    offs_k = tl.arange(0, $(cfg.BK))
    a_ptrs = a_ptr + (offs_am[:, None] * $(cfg.A.sr) + offs_k[None, :] * $(cfg.A.sc))
    b_ptrs = b_ptr + (offs_k[:, None] * $(cfg.B.sr) + offs_bn[None, :] * $(cfg.B.sc))
    accumulator = tl.zeros([$(cfg.BM), $(cfg.BN)], dtype=tl.float32)
    for k in range($(0), $(cfg.steps), $(1)) {
      a = tl.load(a_ptrs, mask=offs_k[None, :] < $(cfg.K) - k * $(cfg.BK), other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < $(cfg.K) - k * $(cfg.BK), other=0.0)
      accumulator = tl.dot(a, b, accumulator)
      a_ptrs += $(cfg.BK) * $(cfg.A.sc)
      b_ptrs += $(cfg.BK) * $(cfg.B.sr)
    }
    c = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * $(cfg.BM) + tl.arange(0, $(cfg.BM))
    offs_cn = pid_n * $(cfg.BN) + tl.arange(0, $(cfg.BN))
    c_ptrs = c_ptr + $(cfg.C.sr) * offs_cm[:, None] + $(cfg.C.sc) * offs_cn[None, :]
    tl.store(c_ptrs, tl.cast(c, tl.float32), mask=(offs_cm[:, None] < $(cfg.M)) & (offs_cn[None, :] < $(cfg.N)))
  }

def remaining (cfg : Config) := typedRemaining "C" "A" "B" cfg

end VeriTile.Bench.Repairs.MatmulKernel.KernelSource
