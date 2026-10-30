import VeriTile.Triton.Core.Reshape
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000

def pairLane (pair : Fin 32) (side : Fin 2) : Fin 64 :=
  ⟨2*pair.val+side.val,by have := pair.isLt; have := side.isLt; omega⟩

/-- Exact finite order of the shared reshape operator. This is a kernel-
checked finite index fact, independent of every floating value and operation. -/
theorem reshape_split : ∀ (row : Fin 16) (pair : Fin 32) (side : Fin 2),
    TileShape.reshapeIndex (input:=[16,64]) (output:=[16,32,2]) (by decide)
      (row,pair,side,PUnit.unit)=(row,pairLane pair side,PUnit.unit) := by
  decide

def pairOfLane (lane : Fin 64) : Fin 32 := ⟨lane.val/2,by have := lane.isLt; omega⟩
def sideOfLane (lane : Fin 64) : Fin 2 := ⟨lane.val%2,by omega⟩

theorem reshape_join : ∀ (row : Fin 16) (lane : Fin 64),
    TileShape.reshapeIndex (input:=[16,32,2]) (output:=[16,64]) (by decide)
      (row,lane,PUnit.unit)=(row,pairOfLane lane,sideOfLane lane,PUnit.unit) := by
  decide

#axiomsClean reshape_split
#axiomsClean reshape_join
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
