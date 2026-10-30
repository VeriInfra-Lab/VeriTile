import bench.tritonbench_g.matmul_triton_autotune.Sources.Mathematical_Host

/-! Canonical checked FP32 matmul source with live dot carry and FP16 output.
Source: matmul_triton_autotune_view_safe.py, SHA256
57ef247c2cd52e4300a270ac83707881b15e11d067dad81430c5d0bde1f75e08.
One readable quotation is shared by both public kernel specifications.
Natural address/control expressions model the explicit signed64 casts; existing
Native proofs establish their exactness from ordinary Tensor/launch bounds. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural
open Whole
noncomputable section

namespace Source
open FPValue
/-- Exact changed register sequence in the candidate. Integer widening is
represented by Nat here; its range proof is separately in Addresses. -/
def typed (p : Parameters) (ACTIVATION : Bool) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let A := p.A
  let B := p.B
  let C := p.C
  triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(p.M), $(BM p))
  num_pid_n = tl.cdiv($(p.N), $(BN p))
  num_pid_in_group = $(p.GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(p.GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(p.GROUP_SIZE_M))
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BM p) + tl.arange(0, $(BM p))) % $(p.M)
  offs_bn = (pid_n * $(BN p) + tl.arange(0, $(BN p))) % $(p.N)
  offs_k = tl.arange(0, $(BK p))
  accumulator = tl.zeros([$(BM p), $(BN p)], dtype=tl.float32)
  for kk in range($(0), $(p.numKBlocks), $(1)) {
    offs_kk = kk * $(BK p) + offs_k
    safe_k = tl.where(offs_kk < $(p.K), offs_kk, $(0))
    a_ptrs = A + offs_am[:, None] * $(p.stride_am) + safe_k[None, :] * $(p.stride_ak)
    b_ptrs = B + safe_k[:, None] * $(p.stride_bk) + offs_bn[None, :] * $(p.stride_bn)
    a = tl.load(a_ptrs, mask=offs_kk[None, :] < $(p.K), other=0.0, dtype=DTYPE)
    b = tl.load(b_ptrs, mask=offs_kk[:, None] < $(p.K), other=0.0, dtype=DTYPE)
    accumulator = tl.dot(a, b, accumulator)
  }
  if ACTIVATION {
    accumulator = tl.where(accumulator >= 0.0, accumulator, 0.01 * accumulator)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM p) + tl.arange(0, $(BM p))
  offs_cn = pid_n * $(BN p) + tl.arange(0, $(BN p))
  safe_cm = tl.where(offs_cm < $(p.M), offs_cm, $(0))
  safe_cn = tl.where(offs_cn < $(p.N), offs_cn, $(0))
  c_ptrs = C + $(p.stride_cm) * safe_cm[:, None] + $(p.stride_cn) * safe_cn[None, :]
  c_mask = (offs_cm[:, None] < $(p.M)) & (offs_cn[None, :] < $(p.N))
  tl.store(c_ptrs, c, mask=c_mask)
}

def kernel (p : Parameters) := typed p p.ACTIVATION

end Source

def sourceKernel (r : Request) : ComputeKernel := Source.kernel r.parameters
abbrev sourceConfigurations := Whole.configurations

end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
