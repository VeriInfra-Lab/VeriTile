import bench.tritonbench_g.matmul_leakyrelu.Kernels
import VeriTile.Triton.Float.Structural

/-! Typed call wrapper for the canonical repaired-source quotation. -/
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000

/-- Compatibility binding to the complete copied Python body in Kernels.lean. -/
def typed (dtype : FloatDType) (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) (ACTIVATION : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype A B C M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M (if ACTIVATION then "leaky_relu" else "")

def source (p : Parameters) (st : Storage) : ComputeKernel :=
  typed (dtype st) p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M p.ACTIVATION
abbrev BM (p : Parameters) := p.BLOCK_SIZE_M
abbrev BN (p : Parameters) := p.BLOCK_SIZE_N
abbrev BK (p : Parameters) := p.BLOCK_SIZE_K
def trips (p : Parameters) := (p.K+BK p-1)/BK p
abbrev AIndex (p : Parameters) := TileIndex [BM p,BK p]
abbrev BIndex (p : Parameters) := TileIndex [BK p,BN p]
abbrev CIndex (p : Parameters) := TileIndex [BM p,BN p]
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
