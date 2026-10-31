import bench.tritonbench_g.matmul_triton1.Sources.Parameters
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("z_ptr",.fp32)]
  scalars := [("m_size",.integer),("k_size",.integer),("n_size",.integer),
    ("stride_xm",.integer),("stride_xk",.integer),("stride_yk",.integer),
    ("stride_yn",.integer),("stride_zm",.integer),("stride_zn",.integer)]
  constexpr := ["m_block_size","k_block_size","n_block_size"]

def body (x_ptr y_ptr z_ptr : RegionName) (m_block_size k_block_size n_block_size : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    m_size = tl.cast(m_size, tl.uint64)
    k_size = tl.cast(k_size, tl.uint64)
    n_size = tl.cast(n_size, tl.uint64)
    stride_xm = tl.cast(stride_xm, tl.uint64)
    stride_xk = tl.cast(stride_xk, tl.uint64)
    stride_yk = tl.cast(stride_yk, tl.uint64)
    stride_yn = tl.cast(stride_yn, tl.uint64)
    stride_zm = tl.cast(stride_zm, tl.uint64)
    stride_zn = tl.cast(stride_zn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks

    m_offsets = tl.arange(0, m_block_size).to(tl.uint64) + m_block * m_block_size
    n_offsets = tl.arange(0, n_block_size).to(tl.uint64) + n_block * n_block_size
    k_offsets = tl.arange(0, k_block_size).to(tl.uint64)

    x_ptrs = x_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
    y_ptrs = y_ptr + k_offsets[:, None] * stride_yk + n_offsets[None, :] * stride_yn
    z_ptrs = z_ptr + m_offsets[:, None] * stride_zm + n_offsets[None, :] * stride_zn

    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    for k in range(0, tl.cdiv(k_size, k_block_size)):
        x_sub = tl.load(x_ptrs, mask=(m_offsets[:, None] < m_size) & (k_offsets[None, :] < k_size - k * k_block_size), other=0.0)
        y_sub = tl.load(y_ptrs, mask=(k_offsets[:, None] < k_size - k * k_block_size) & (n_offsets[None, :] < n_size), other=0.0)
        z += tl.dot(x_sub, y_sub, allow_tf32=False)
        x_ptrs += k_block_size * stride_xk
        y_ptrs += k_block_size * stride_yk

    tl.store(z_ptrs, z, mask=(m_offsets[:, None] < m_size) & (n_offsets[None, :] < n_size))
  }

def scalarParameters (p : Core.Parameters) : List (RegName × Nat) :=
  [("m_size",p.rows),("k_size",p.innerSize),("n_size",p.columns),
    ("stride_xm",p.inputRow),("stride_xk",p.inputInner),("stride_yk",p.weightInner),
    ("stride_yn",p.weightColumn),("stride_zm",p.outputRow),("stride_zn",p.outputColumn)]
noncomputable def arguments (p : Core.Parameters) : List ScalarArgument :=
  [ScalarArgument.integer "m_size" p.rows,ScalarArgument.integer "k_size" p.innerSize,
    ScalarArgument.integer "n_size" p.columns,ScalarArgument.integer "stride_xm" p.inputRow,
    ScalarArgument.integer "stride_xk" p.inputInner,ScalarArgument.integer "stride_yk" p.weightInner,
    ScalarArgument.integer "stride_yn" p.weightColumn,ScalarArgument.integer "stride_zm" p.outputRow,
    ScalarArgument.integer "stride_zn" p.outputColumn]
noncomputable def source (p : Core.Parameters) : ComputeKernel :=
  (body p.input p.weight p.output p.blockRows p.blockInner p.blockColumns).withScalarArguments (arguments p)

/-- Cache the row/column comparisons without changing the ordered FP dot. -/
def maskHoistedBody (x_ptr y_ptr z_ptr : RegionName) (m_block_size k_block_size n_block_size : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    m_size = tl.cast(m_size, tl.uint64)
    k_size = tl.cast(k_size, tl.uint64)
    n_size = tl.cast(n_size, tl.uint64)
    stride_xm = tl.cast(stride_xm, tl.uint64)
    stride_xk = tl.cast(stride_xk, tl.uint64)
    stride_yk = tl.cast(stride_yk, tl.uint64)
    stride_yn = tl.cast(stride_yn, tl.uint64)
    stride_zm = tl.cast(stride_zm, tl.uint64)
    stride_zn = tl.cast(stride_zn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks
    m_offsets = tl.arange(0, m_block_size).to(tl.uint64) + m_block * m_block_size
    n_offsets = tl.arange(0, n_block_size).to(tl.uint64) + n_block * n_block_size
    k_offsets = tl.arange(0, k_block_size).to(tl.uint64)
    row_mask = m_offsets < m_size
    column_mask = n_offsets < n_size
    x_ptrs = x_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
    y_ptrs = y_ptr + k_offsets[:, None] * stride_yk + n_offsets[None, :] * stride_yn
    z_ptrs = z_ptr + m_offsets[:, None] * stride_zm + n_offsets[None, :] * stride_zn
    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)
    for k in range(0, tl.cdiv(k_size, k_block_size)) {
      x_sub = tl.load(x_ptrs, mask=row_mask[:, None] & (k_offsets[None, :] < k_size - k * k_block_size), other=0.0)
      y_sub = tl.load(y_ptrs, mask=(k_offsets[:, None] < k_size - k * k_block_size) & column_mask[None, :], other=0.0)
      z += tl.dot(x_sub, y_sub, allow_tf32=false)
      x_ptrs += k_block_size * stride_xk
      y_ptrs += k_block_size * stride_yk
    }
    tl.store(z_ptrs, z, mask=row_mask[:, None] & column_mask[None, :])
  }

noncomputable def maskHoisted (p : Core.Parameters) : ComputeKernel :=
  (maskHoistedBody p.input p.weight p.output p.blockRows p.blockInner p.blockColumns).withScalarArguments (arguments p)

end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
