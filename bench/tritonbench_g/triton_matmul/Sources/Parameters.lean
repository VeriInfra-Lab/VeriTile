import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton
inductive Storage where | half | e4m3
  deriving DecidableEq

def dtype : Storage → FloatDType | .half => .fp16 | .e4m3 => .f8e4
structure Parameters where
  (A B C : RegionName)
  (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
abbrev BM (p : Parameters) := p.BLOCK_SIZE_M
abbrev BN (p : Parameters) := p.BLOCK_SIZE_N
abbrev BK (p : Parameters) := p.BLOCK_SIZE_K
abbrev AIndex (p : Parameters) := TileIndex [BM p,BK p]
abbrev BIndex (p : Parameters) := TileIndex [BK p,BN p]
abbrev CIndex (p : Parameters) := TileIndex [BM p,BN p]

end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
