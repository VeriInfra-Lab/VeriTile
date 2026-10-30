import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.mul_exponent_compensator.Kernels

/-! Exact selected repaired Python bodies, including the static constexpr
power and both signed-int64 index casts. Only BLOCK_SIZE is a call constexpr;
the runtime length uses the existing mathematical index projection, with its
original native signed bounds retained in Proofs/Indices. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
open VeriTile.Triton
noncomputable section

abbrev exponentCompensator := VeriTile.Bench.TritonBenchG.MulExponentCompensator.exponentCompensator
abbrev Word := BitVec 32
def constantWord : Word := 0x77800000
def grid (n B : Nat) : Nat := (n+B-1)/B

def sourceSignature : CallSignature where
  pointers := [("src",.fp32),("dst",.fp32)]
  scalars := [("n_elements",.i64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["src"]
  writePorts := ["dst"]

def sourceBody (src dst : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args [n_elements := n_elements] in
  triton {
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    # Calculate the indices for the current program ID
    idxs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = idxs < n_elements
    # Load elements from the source tensor
    x = tl.load(src + idxs, mask=mask)
    # Multiply each element by the exponent compensator
    y = x * exponent_compensator
    # Store the result in the destination tensor
    tl.store(dst + idxs, y, mask=mask)
  }

def scalarLeftBody (src dst : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args [n_elements := n_elements] in
  triton {
    """Existing scalar-left variant with the same widened masked accesses."""
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    idxs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = idxs < n_elements
    x = tl.load(src + idxs, mask=mask)
    y = exponent_compensator * x
    tl.store(dst + idxs, y, mask=mask)
  }

def candidateKernel (src dst : RegionName) (n B : Nat) : ComputeKernel :=
  sourceBody src dst n B

def scalarLeftKernel (src dst : RegionName) (n B : Nat) : ComputeKernel :=
  scalarLeftBody src dst n B

end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
