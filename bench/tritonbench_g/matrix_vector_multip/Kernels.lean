import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels
open VeriTile.Triton

/-- The checked contiguous FP32 source, shared by Real and native-graph
interpretations. Integer coordinates here are mathematical naturals; the
uint64 casts and unsigned while in Python are justified by KernelIndex.
The complete loop, initial +0, load casts and output cast remain explicit. -/
def source (N M BLOCK_N BLOCK_M : Nat) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let A : RegionName := "A"
  let B : RegionName := "B"
  let C : RegionName := "C"
  triton {
    tl.region A = DTYPE, B = DTYPE, C = DTYPE
    pid = tl.program_id(0)
    offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))[:, None]
    offset_m = tl.arange(0, $(BLOCK_M))[None, :]
    n_mask = offset_n < $(N)
    A_ptrs = A + offset_n * $(M) + offset_m * $(1)
    B_ptrs = B + offset_m * $(1)
    acc = tl.zeros([$(BLOCK_N), $(BLOCK_M)], dtype=tl.float32)
    for m in range(0, $(M), $(BLOCK_M)) {
      m_mask = (m + offset_m) < $(M)
      a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
      b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
      acc += a * b
      A_ptrs += $(BLOCK_M) * $(1)
      B_ptrs += $(BLOCK_M) * $(1)
    }
    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * $(1)
    tl.store(C_ptrs, (acc[:, None]).to(DTYPE), mask=n_mask)
  }

/-- Explicit FMA candidate for the selected source's already fused native
carry graph. It fixes fusion at the source level and avoids a separate product
temporary; loads, casts, loop order, initial +0 and reduction are retained. -/
def explicitFMA (N M BLOCK_N BLOCK_M : Nat) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let A : RegionName := "A"
  let B : RegionName := "B"
  let C : RegionName := "C"
  triton {
    tl.region A = DTYPE, B = DTYPE, C = DTYPE
    pid = tl.program_id(0)
    offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))[:, None]
    offset_m = tl.arange(0, $(BLOCK_M))[None, :]
    n_mask = offset_n < $(N)
    A_ptrs = A + offset_n * $(M) + offset_m * $(1)
    B_ptrs = B + offset_m * $(1)
    acc = tl.zeros([$(BLOCK_N), $(BLOCK_M)], dtype=tl.float32)
    for m in range(0, $(M), $(BLOCK_M)) {
      m_mask = (m + offset_m) < $(M)
      a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
      b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
      acc = tl.fma(a, b, acc)
      A_ptrs += $(BLOCK_M) * $(1)
      B_ptrs += $(BLOCK_M) * $(1)
    }
    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * $(1)
    tl.store(C_ptrs, (acc[:, None]).to(DTYPE), mask=n_mask)
  }

end VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels
