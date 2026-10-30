import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels
open VeriTile.Triton

/-- Caller metadata for the existing contiguous FP32 view. The quoted bodies
retain their explicit unsigned casts and while-loop counter. Host integer values enter before the six source uint64 casts;
representability is established by the source execution contract. -/
def sourceSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("C",.fp32)]
  scalars := [("N",.integer),("M",.integer),("stride_an",.integer),("stride_am",.integer),
    ("stride_bm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_N","BLOCK_M"]
  readPorts := ["A","B"]
  writePorts := ["C"]
  typedFP32Storage := Bool.true

def source (N M BLOCK_N BLOCK_M : Nat)
    (rowShape : Random.ArangeSizeDomain BLOCK_N)
    (columnShape : Random.ArangeSizeDomain BLOCK_M) : ComputeKernel :=
  let BLOCK_N := Random.i32ArangeSize BLOCK_N rowShape
  let BLOCK_M := Random.i32ArangeSize BLOCK_M columnShape
  let A : RegionName := "A"
  let B : RegionName := "B"
  let C : RegionName := "C"
  let stride_an : Nat := M
  let stride_am : Nat := 1
  let stride_bm : Nat := 1
  let stride_cn : Nat := 1
  with_triton_call sourceSignature call_args
    [N := N,M := M,stride_an := stride_an,stride_am := stride_am,
     stride_bm := stride_bm,stride_cn := stride_cn] in
  triton {
    N = tl.cast(N, tl.uint64)
    M = tl.cast(M, tl.uint64)
    stride_an = tl.cast(stride_an, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_bm = tl.cast(stride_bm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.uint64)[:, None]
    offset_m = tl.arange(0, BLOCK_M).to(tl.uint64)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    m = tl.full((), 0, tl.uint64)
    while m < M:
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm
        m += BLOCK_M

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)
  }

def explicitFMA (N M BLOCK_N BLOCK_M : Nat)
    (rowShape : Random.ArangeSizeDomain BLOCK_N)
    (columnShape : Random.ArangeSizeDomain BLOCK_M) : ComputeKernel :=
  let BLOCK_N := Random.i32ArangeSize BLOCK_N rowShape
  let BLOCK_M := Random.i32ArangeSize BLOCK_M columnShape
  let A : RegionName := "A"
  let B : RegionName := "B"
  let C : RegionName := "C"
  let stride_an : Nat := M
  let stride_am : Nat := 1
  let stride_bm : Nat := 1
  let stride_cn : Nat := 1
  with_triton_call sourceSignature call_args
    [N := N,M := M,stride_an := stride_an,stride_am := stride_am,
     stride_bm := stride_bm,stride_cn := stride_cn] in
  triton {
    N = tl.cast(N, tl.uint64)
    M = tl.cast(M, tl.uint64)
    stride_an = tl.cast(stride_an, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_bm = tl.cast(stride_bm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.uint64)[:, None]
    offset_m = tl.arange(0, BLOCK_M).to(tl.uint64)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    m = tl.full((), 0, tl.uint64)
    while m < M:
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc = tl.fma(a, b, acc)
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm
        m += BLOCK_M

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)
  }

end VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels
