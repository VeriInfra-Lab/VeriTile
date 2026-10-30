import bench.tritonbench_g.matmul_triton1.Sources.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Shared readable source entry. `source` and its candidates retain the
selected compiler FP32 accumulator-dot graph. `pythonBody` and `pythonSource`
retain the original standalone dot, following addition, and uint64 casts.
The existing proofs cover the selected compiler graph; retaining both sources
does not establish their FP equivalence. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton
def source (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
/-- Constant-first row-block index canonicalization; complete source retained. -/
def rowCanonical (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = $(p.blockRows) * pid_m + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_am[:, None] < $(p.rows)) & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & (offs_bn[None, :] < $(p.columns)), other=0.0)
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
/-- Cache loop-invariant row/column comparisons for all K loads and the final store. -/
def maskHoisted (p : Revision001.Core.Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = tl.float32, weight = tl.float32, output = tl.float32
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  row_mask = offs_am < $(p.rows)
  column_mask = offs_bn < $(p.columns)
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(p.innerSize), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=row_mask[:, None] & (offs_k[None, :] < $(p.innerSize) - k * $(p.blockInner)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(p.innerSize) - k * $(p.blockInner)) & column_mask[None, :], other=0.0)
    accumulator = tl.dot(a, b, accumulator, allow_tf32=false)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = row_mask[:, None] & column_mask[None, :]
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton
/-- Source-level interface; runtime integers are explicitly cast to uint64
by the body. This interface does not certify a native launch or compiler pass. -/
def pythonCallSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("z_ptr",.fp32)]
  scalars := [("m_size",.integer),("k_size",.integer),("n_size",.integer),("stride_xm",.integer),("stride_xk",.integer),("stride_yk",.integer),("stride_yn",.integer),("stride_zm",.integer),("stride_zn",.integer)]
  constexpr := ["m_block_size","k_block_size","n_block_size"]

/-- Complete Python-source arithmetic and uint64 casts. Its standalone dot
and following addition are distinct from the selected compiler graph above.
The existing Correct/FPEquiv entries continue to concern that selected graph. -/
def pythonBody (x_ptr y_ptr z_ptr : RegionName) (m_block_size k_block_size n_block_size : Nat) : ComputeKernel :=
  with_triton_signature pythonCallSignature in
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

    z = tl.zeros([m_block_size, n_block_size], dtype=tl.float32)

    for k in range($(0), tl.cdiv(k_size, k_block_size)) {
        x_sub = tl.load(x_ptrs, mask=(m_offsets[:, None] < m_size) & (k_offsets[None, :] < k_size - k * k_block_size), other=0.0)
        y_sub = tl.load(y_ptrs, mask=(k_offsets[:, None] < k_size - k * k_block_size) & (n_offsets[None, :] < n_size), other=0.0)
        z += tl.dot(x_sub, y_sub, allow_tf32=False)
        x_ptrs += k_block_size * stride_xk
        y_ptrs += k_block_size * stride_yk
    }

    tl.store(z_ptrs, z, mask=(m_offsets[:, None] < m_size) & (n_offsets[None, :] < n_size))
  }

/-- Call-site values are bound separately, before executing the source body. -/
def pythonArguments (p : Revision001.Core.Parameters) : List ScalarArgument :=
  [ScalarArgument.integer "m_size" (Int.ofNat p.rows),
   ScalarArgument.integer "k_size" (Int.ofNat p.innerSize),
   ScalarArgument.integer "n_size" (Int.ofNat p.columns),
   ScalarArgument.integer "stride_xm" (Int.ofNat p.inputRow),
   ScalarArgument.integer "stride_xk" (Int.ofNat p.inputInner),
   ScalarArgument.integer "stride_yk" (Int.ofNat p.weightInner),
   ScalarArgument.integer "stride_yn" (Int.ofNat p.weightColumn),
   ScalarArgument.integer "stride_zm" (Int.ofNat p.outputRow),
   ScalarArgument.integer "stride_zn" (Int.ofNat p.outputColumn)]

def pythonSource (p : Revision001.Core.Parameters) : ComputeKernel :=
  (pythonBody p.input p.weight p.output p.blockRows p.blockInner p.blockColumns).withScalarArguments (pythonArguments p)
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
