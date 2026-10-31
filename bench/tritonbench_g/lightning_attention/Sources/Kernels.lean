import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile.Triton
structure Parameters where
  (B H N D E BT BM NT CB NC : Nat)

end VeriTile.Bench.Repairs.LightningAttention
