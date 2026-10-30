import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.batched_vecmat_mult.Kernels

/-! Complete selected int64 Python function bodies. Storage, runtime dimension
bindings and constexpr blocks live outside each quotation. The authored Python
counterpart preserves the existing block-first carry-addition candidate. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
abbrev Parameters := ValueProofs.Parameters

def callSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("output",.fp32)]
  scalars := [("dim_m",.integer),("dim_n",.integer),("dim_k",.integer)]
  constexpr := ["block_m","block_n","block_k"]
  readPorts := ["A","B"]
  writePorts := ["output"]

def kernel (A B output : RegionName)
    (dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) : ComputeKernel :=
  let block_m := BLOCK_M
  let block_n := BLOCK_N
  let block_k := BLOCK_K
  with_triton_call callSignature call_args [dim_m := dim_m, dim_n := dim_n, dim_k := dim_k] in triton {
    dim_n = tl.cast(dim_n, tl.int64)
    dim_k = tl.cast(dim_k, tl.int64)
    m_index = tl.program_id(0).to(tl.int64)
    n_index = tl.program_id(1).to(tl.int64)
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

    tl.store(output + output_tile, vecmat)
  }

def blockFirstKernel (A B output : RegionName)
    (dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) : ComputeKernel :=
  let block_m := BLOCK_M
  let block_n := BLOCK_N
  let block_k := BLOCK_K
  with_triton_call callSignature call_args [dim_m := dim_m, dim_n := dim_n, dim_k := dim_k] in triton {
    dim_n = tl.cast(dim_n, tl.int64)
    dim_k = tl.cast(dim_k, tl.int64)
    m_index = tl.program_id(0).to(tl.int64)
    n_index = tl.program_id(1).to(tl.int64)
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat = tl.trans(tl.sum(expanded_a * b, axis=2)) + vecmat

    tl.store(output + output_tile, vecmat)
  }

def source (p : Parameters) : ComputeKernel :=
  kernel p.A p.B p.output p.M p.N p.K p.BM p.BN p.BK

def blockFirst (p : Parameters) : ComputeKernel :=
  blockFirstKernel p.A p.B p.output p.M p.N p.K p.BM p.BN p.BK

end VeriTile.Bench.Repairs.BatchedVecmatMult
