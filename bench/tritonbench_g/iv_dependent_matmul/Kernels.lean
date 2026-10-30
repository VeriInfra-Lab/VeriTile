import bench.tritonbench_g.iv_dependent_matmul.Contract

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape
open Lean VeriTile.Triton
open TritonBenchG.IvDependentMatmul FPValue
namespace Kernel

/-! Complete ordinary quotations for each retained pointer schedule and graph.
The source graph has Dot+Add; the compiled graph has the existing live carry
operand. These are preserved program views, not a numerical rewrite identity.
All signed64/pointer correspondence remains proved at its original boundary. -/

def preLoadSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a_ptrs = a_ptr + k * $(BK) * $(sak)
    b_ptrs = b_ptr + k * $(BK) * $(sbk)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator += tl.dot(a, b)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def postLoadSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs = a_ptr + (k + $(1)) * $(BK) * $(sak)
    b_ptrs = b_ptr + (k + $(1)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def mixedSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a_ptrs = a_ptr + k * $(BK) * $(sak)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator += tl.dot(a, b)
    b_ptrs = b_ptr + (k + $(1)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def twoSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  a_ptrs_next = a_ptr + $(1) * $(BK) * $(sak)
  b_ptrs_next = b_ptr + $(1) * $(BK) * $(sbk)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs = a_ptrs_next
    b_ptrs = b_ptrs_next
    a_ptrs_next = a_ptr + (k + $(2)) * $(BK) * $(sak)
    b_ptrs_next = b_ptr + (k + $(2)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def threeSource (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  a_ptrs_next = a_ptr + $(1) * $(BK) * $(sak)
  b_ptrs_next = b_ptr + $(1) * $(BK) * $(sbk)
  a_ptrs_next_next = a_ptr + $(2) * $(BK) * $(sak)
  b_ptrs_next_next = b_ptr + $(2) * $(BK) * $(sbk)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs = a_ptrs_next
    b_ptrs = b_ptrs_next
    a_ptrs_next = a_ptrs_next_next
    b_ptrs_next = b_ptrs_next_next
    a_ptrs_next_next = a_ptr + (k + $(3)) * $(BK) * $(sak)
    b_ptrs_next_next = b_ptr + (k + $(3)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def preLoadNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a_ptrs = a_ptr + k * $(BK) * $(sak)
    b_ptrs = b_ptr + k * $(BK) * $(sbk)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def postLoadNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs = a_ptr + (k + $(1)) * $(BK) * $(sak)
    b_ptrs = b_ptr + (k + $(1)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def mixedNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a_ptrs = a_ptr + k * $(BK) * $(sak)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    b_ptrs = b_ptr + (k + $(1)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def twoNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  a_ptrs_next = a_ptr + $(1) * $(BK) * $(sak)
  b_ptrs_next = b_ptr + $(1) * $(BK) * $(sbk)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs = a_ptrs_next
    b_ptrs = b_ptrs_next
    a_ptrs_next = a_ptr + (k + $(2)) * $(BK) * $(sak)
    b_ptrs_next = b_ptr + (k + $(2)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def threeNative (dtype : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BN))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptr = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  b_ptr = B + offs_k[:, None] * $(sbk) + offs_bn[None, :] * $(sbn)
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  a_ptrs_next = a_ptr + $(1) * $(BK) * $(sak)
  b_ptrs_next = b_ptr + $(1) * $(BK) * $(sbk)
  a_ptrs_next_next = a_ptr + $(2) * $(BK) * $(sak)
  b_ptrs_next_next = b_ptr + $(2) * $(BK) * $(sbk)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BK), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BK), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs = a_ptrs_next
    b_ptrs = b_ptrs_next
    a_ptrs_next = a_ptrs_next_next
    b_ptrs_next = b_ptrs_next_next
    a_ptrs_next_next = a_ptr + (k + $(3)) * $(BK) * $(sak)
    b_ptrs_next_next = b_ptr + (k + $(3)) * $(BK) * $(sbk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_cn = pid_n * $(BN) + tl.arange(0, $(BN))
  c_ptrs = C + $(scm) * offs_cm[:, None] + $(scn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def quotation (s : Kernels.Schedule) (g : Graph) (p : Parameters) : ComputeKernel :=
  (match s, g with
  | .preLoad, .source => preLoadSource
  | .postLoad, .source => postLoadSource
  | .mixed, .source => mixedSource
  | .two, .source => twoSource
  | .three, .source => threeSource
  | .preLoad, .accumulator => preLoadNative
  | .postLoad, .accumulator => postLoadNative
  | .mixed, .accumulator => mixedNative
  | .two, .accumulator => twoNative
  | .three, .accumulator => threeNative)
    .fp32 p.input p.weight p.output p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
    p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
    p.blockRows p.blockColumns p.blockInner

def source (r : Whole.Request) := quotation r.schedule .source r.parameters
def compiled (r : Whole.Request) := quotation r.schedule .accumulator r.parameters

end Kernel
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape
