import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Contract
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open VeriTile.Triton FPValue Host
noncomputable section

abbrev params := StaticChecked.params

def callSignature : CallSignature where
  pointers := [("a_ptr",.inputFloat),("b_ptr",.inputFloat),("c_ptr",.outputFloat)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),
    ("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),
    ("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","BLOCK_K","OUTPUT_F16"]

def source (p : Parameters) (half : Bool) : ComputeKernel :=
  let dtype : FloatDType := .fp16
  let out_dtype := outputDType half
  let a_ptr := p.a
  let b_ptr := p.b
  let c_ptr := p.c
  let BLOCK_M := p.blockRows
  let BLOCK_N := p.blockColumns
  let BLOCK_K := p.blockInner
  let OUTPUT_F16 := half
  with_triton_call callSignature call_args [M := p.rows, N := p.columns, K := p.inner,
    stride_am := p.aRow, stride_ak := p.aInner, stride_bk := p.bInner,
    stride_bn := p.bColumn, stride_cm := p.cRow, stride_cn := p.cColumn] in triton {
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    # Load A and B blocks
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    # Compute matrix product
    c = tl.dot(a, b)
    # Optionally convert the result to float16
    if OUTPUT_F16:
        c = c.to(tl.float16)

    # Store the result
    tl.store(c_block_ptr, c)
  }

def sourceExplicitOutput (p : Parameters) (half : Bool) : ComputeKernel :=
  let dtype : FloatDType := .fp16
  let out_dtype := outputDType half
  let a_ptr := p.a
  let b_ptr := p.b
  let c_ptr := p.c
  let BLOCK_M := p.blockRows
  let BLOCK_N := p.blockColumns
  let BLOCK_K := p.blockInner
  let OUTPUT_F16 := half
  with_triton_call callSignature call_args [M := p.rows, N := p.columns, K := p.inner,
    stride_am := p.aRow, stride_ak := p.aInner, stride_bk := p.bInner,
    stride_bn := p.bColumn, stride_cm := p.cRow, stride_cn := p.cColumn] in triton {
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    # Load A and B blocks
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    # Compute matrix product
    c = tl.dot(a, b)
    # Optionally convert the result to float16
    if OUTPUT_F16:
        c = c.to(tl.float16)
    else:
        c = c.to(c_ptr.type.element_ty)

    # Store the result
    tl.store(c_block_ptr, c)
  }

def kernel (h : Shape) (f : Flags) : ComputeKernel := source (params h f) f.half
def kernelExplicitOutput (h : Shape) (f : Flags) : ComputeKernel := sourceExplicitOutput (params h f) f.half

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
