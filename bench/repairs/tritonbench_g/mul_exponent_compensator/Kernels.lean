import bench.tritonbench_g.mul_exponent_compensator.Kernels

/-! Separate complete masked scaling source. Production storage/load/multiply/
store is FP32. The original constexpr and src*constant order are retained.
Native i64 refinement and the host's actual user block are proved separately. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
open VeriTile.Triton
noncomputable section

abbrev exponentCompensator := VeriTile.Bench.TritonBenchG.MulExponentCompensator.exponentCompensator
abbrev Word := BitVec 32
def constantWord : Word := 0x77800000
def grid (n B : Nat) : Nat := (n+B-1)/B

/-- The selected FP32 view of the actual repaired JIT body. Integer controls
are the Nat projection of the independently checked widened native operations. -/
def candidateKernel (src dst : RegionName) (n B : Nat) : ComputeKernel := triton {
  tl.region src = tl.float32, dst = tl.float32
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(B) + tl.arange(0, $(B))
  mask = idxs < $(n)
  x = tl.load(src + idxs, mask=mask)
  y = x * exponent_compensator
  tl.store(dst + idxs, y, mask=mask)
}

/-- The existing scalar-left multiplication candidate, applied to the complete
repaired masked source. Only the FP32 multiplication operands are exchanged. -/
def scalarLeftKernel (src dst : RegionName) (n B : Nat) : ComputeKernel := triton {
  tl.region src = tl.float32, dst = tl.float32
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(B) + tl.arange(0, $(B))
  mask = idxs < $(n)
  x = tl.load(src + idxs, mask=mask)
  y = exponent_compensator * x
  tl.store(dst + idxs, y, mask=mask)
}

end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
