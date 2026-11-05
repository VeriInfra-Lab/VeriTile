import VeriTile.Triton.DSL
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource
open VeriTile.Triton
inductive Storage where | fp16 | bf16 deriving DecidableEq
def Storage.float : Storage → FloatDType | .fp16=>.fp16 | .bf16=>.bf16
def Storage.tile (s : Storage) : TileDType := s.float.toTileDType
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource
