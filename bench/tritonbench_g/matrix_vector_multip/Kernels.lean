import bench.tritonbench_g.matrix_vector_multip.Sources.Parameters
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("C",.fp32)]
  scalars := [("N",.integer),("M",.integer),("stride_an",.integer),
    ("stride_am",.integer),("stride_bm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_N","BLOCK_M"]

def body (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
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

def fmaBody (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
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

noncomputable def arguments (p : VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.Parameters) : List ScalarArgument :=
  [ScalarArgument.integer "N" p.rows,ScalarArgument.integer "M" p.columns,
    ScalarArgument.integer "stride_an" p.strideRow,ScalarArgument.integer "stride_am" p.strideColumn,
    ScalarArgument.integer "stride_bm" p.strideVector,ScalarArgument.integer "stride_cn" p.strideOut]

noncomputable def fromParameters (p : VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.Parameters) : ComputeKernel :=
  (body p.a p.b p.out p.blockRows p.blockColumns).withScalarArguments (arguments p)
noncomputable def fmaFromParameters (p : VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.Parameters) : ComputeKernel :=
  (fmaBody p.a p.b p.out p.blockRows p.blockColumns).withScalarArguments (arguments p)
def parameters (N M BN BM : Nat) : VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.Parameters :=
  ⟨"A","B","C",N,M,M,1,1,1,BN,BM⟩
noncomputable def source (N M BN BM : Nat) : ComputeKernel := fromParameters (parameters N M BN BM)
noncomputable def explicitFMA (N M BN BM : Nat) : ComputeKernel := fmaFromParameters (parameters N M BN BM)
end VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels
