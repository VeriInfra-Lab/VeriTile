import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Core.SignedCast
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ChunkRetention
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000
private theorem signed_lane_erasure (width n : Nat) :
    (Op.intToSigned width (.castNatToInt (.arange n))).eraseDType =
      Op.intToSigned width (.castNatToInt (.arange n)) := by
  simp [Op.intToSigned,Op.unsignedToInt,Op.intToUnsigned,Op.natToUnsigned,
    Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    IntegralDType.eraseDType,ComparableDType.eraseDType]

theorem signed32_lane_erasure (n : Nat) :
    (Op.intToSigned 32 (.castNatToInt (.arange n))).eraseDType =
      Op.intToSigned 32 (.castNatToInt (.arange n)) := signed_lane_erasure 32 n
#axiomsClean signed32_lane_erasure
end VeriTile.Bench.TritonBenchG.ChunkRetention
