import bench.tritonbench_g.mean_reduction.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Complete original and existing candidate Python function bodies. Runtime
M/N are external call arguments; only BLOCK_M/BLOCK_N are constexpr. The
historical storage family is retained by the import and checked by complete
kernel equalities in Proofs/SourcePreservation. -/
namespace VeriTile.Bench.Repairs.MeanReduction
open VeriTile.Triton

def sourceSignature : CallSignature where
  pointers := [("X",.fp32),("Mean",.fp32)]
  scalars := [("M",.integer),("N",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]
  readPorts := ["X"]
  writePorts := ["Mean"]

def sourceKernel (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args [M := M, N := N] in
  triton {
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)
  }

def indexCommutedSource (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args [M := M, N := N] in
  triton {
    pid = BLOCK_M * tl.program_id(0) + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)
  }

def gridSize (rows : Nat) : Nat := (rows+7)/8

def kernel (M N : Nat) : ComputeKernel := sourceKernel "X" "Mean" M N 8 8

def indexCommutedKernel (M N : Nat) : ComputeKernel := indexCommutedSource "X" "Mean" M N 8 8

end VeriTile.Bench.Repairs.MeanReduction
