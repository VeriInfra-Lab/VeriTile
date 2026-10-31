import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.TypedSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false
inductive Storage where | fp16 | bf16 deriving DecidableEq
 def Storage.float : Storage → FloatDType | .fp16=>.fp16 | .bf16=>.bf16
 def Storage.tile (s : Storage) : TileDType := s.float.toTileDType
namespace FP16

end FP16
namespace BF16

end BF16
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.TypedSource
