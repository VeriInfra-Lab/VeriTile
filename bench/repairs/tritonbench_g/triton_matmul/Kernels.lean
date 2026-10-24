import bench.tritonbench_g.triton_matmul.Kernels

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
noncomputable section

namespace Source
def typed (p : Parameters) (st : Storage) : ComputeKernel :=
  let dtype := dtype st
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
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BM p)
  start_n = pid_n * $(BN p)
  lanes_m = tl.arange(0, $(BM p))
  lanes_n = tl.arange(0, $(BN p))
  mask_m = lanes_m < $(p.M) - start_m
  mask_n = lanes_n < $(p.N) - start_n
  offs_am = tl.where(mask_m, start_m, $(0)) + tl.where(mask_m, lanes_m, $(0))
  offs_bn = tl.where(mask_n, start_n, $(0)) + tl.where(mask_n, lanes_n, $(0))
  offs_k = tl.arange(0, $(BK p))
  accumulator = tl.zeros([$(BM p), $(BN p)], dtype=tl.float32)
  for kk in range($(0), $(p.numKBlocks), $(1)) {
    base_k = kk * $(BK p)
    mask_k = offs_k < $(p.K) - base_k
    safe_k = tl.where(mask_k, base_k, $(0)) + tl.where(mask_k, offs_k, $(0))
    a_ptrs = A + offs_am[:, None] * $(p.stride_am) + safe_k[None, :] * $(p.stride_ak)
    b_ptrs = B + safe_k[:, None] * $(p.stride_bk) + offs_bn[None, :] * $(p.stride_bn)
    a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0, dtype=DTYPE)
    b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0, dtype=DTYPE)
    accumulator = tl.dot(a, b, accumulator)
  }
  c = (accumulator).to(DTYPE)
  c_ptrs = C + $(p.stride_cm) * offs_am[:, None] + $(p.stride_cn) * offs_bn[None, :]
  c_mask = mask_m[:, None] & mask_n[None, :]
  tl.store(c_ptrs, c, mask=c_mask)
}
end Source

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
