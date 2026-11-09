import VeriTile.Triton
import VeriTile.Triton.DSL.PythonCallSyntax
import bench.tritonbench_g.matmul_tma.Proofs.Sources.Parameters

/-! Pasted one-tile Python source and its direct-store conversion candidate.
All pointer storage, runtime dimensions/strides and constexpr tile/output
arguments are supplied outside the body. No synthetic FP32 cast is added. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton
noncomputable section

def sourceSignature (dtype out_dtype : FloatDType) : CallSignature where
  pointers := []
  floatingPointers := [("a_ptr",dtype),("b_ptr",dtype),("c_ptr",out_dtype)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),
    ("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),
    ("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","BLOCK_K","OUTPUT_F16"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def typedKernel (dtype out_dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let a_ptr := p.a
  let b_ptr := p.b
  let c_ptr := p.c
  let M := p.rows
  let N := p.columns
  let K := p.inner
  let stride_am := p.aRow
  let stride_ak := p.aInner
  let stride_bk := p.bInner
  let stride_bn := p.bColumn
  let stride_cm := p.cRow
  let stride_cn := p.cColumn
  let BLOCK_M := p.blockRows
  let BLOCK_N := p.blockColumns
  let BLOCK_K := p.blockInner
  let OUTPUT_F16 := decide (out_dtype = .fp16)
  with_triton_call (sourceSignature dtype out_dtype) call_args
    [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak,
     stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm,
     stride_cn := stride_cn] in triton {
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

def directOutputTypedKernel (dtype out_dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let a_ptr := p.a
  let b_ptr := p.b
  let c_ptr := p.c
  let M := p.rows
  let N := p.columns
  let K := p.inner
  let stride_am := p.aRow
  let stride_ak := p.aInner
  let stride_bk := p.bInner
  let stride_bn := p.bColumn
  let stride_cm := p.cRow
  let stride_cn := p.cColumn
  let BLOCK_M := p.blockRows
  let BLOCK_N := p.blockColumns
  let BLOCK_K := p.blockInner
  let OUTPUT_F16 := decide (out_dtype = .fp16)
  with_triton_call (sourceSignature dtype out_dtype) call_args
    [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak,
     stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm,
     stride_cn := stride_cn] in triton {
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

    # Store the result
    if OUTPUT_F16:
        tl.store(c_block_ptr, c.to(tl.float16))
    else:
        tl.store(c_block_ptr, c)
  }

def outputDType (half : Bool) : FloatDType := if half then .fp16 else .fp32

def source (p : Parameters) (half : Bool) : ComputeKernel :=
  typedKernel .fp16 (outputDType half) p

end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open VeriTile.Triton FPValue
noncomputable section

def sourceDirectOutput (p : Parameters) (half : Bool) : ComputeKernel :=
  directOutputTypedKernel .fp16 (outputDType half) p

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
