import bench.tritonbench_g.chunk_gla_simple.Sources.FP
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

/-- The three actual FP32 dot sites need their native backend to be available.
The result functions are opaque: no value, reduction order or numerical law is
specified. These local capabilities cannot assert whole-kernel success. -/
structure Dots {α : Type} (M : Algebra α) (c : Shape) where
  qh : Values α .real [c.BT,c.BK] → Values α .real [c.BK,c.BV] → Values α .real [c.BT,c.BV]
  qk : Values α .real [c.BT,c.BK] → Values α .real [c.BK,c.BT] → Values α .real [c.BT,c.BT]
  sv : Values α .real [c.BT,c.BT] → Values α .real [c.BT,c.BV] → Values α .real [c.BT,c.BV]
  qh_eq : ∀ a b, M.dotWithOptions (some .fp32) {allowTF32 := some Bool.false} (batch := []) a b = some (qh a b)
  qk_eq : ∀ a b, M.dotWithOptions (some .fp32) {allowTF32 := some Bool.false} (batch := []) a b = some (qk a b)
  sv_eq : ∀ a b, M.dotWithOptions (some .fp32) {allowTF32 := some Bool.false} (batch := []) a b = some (sv a b)

def Present {α : Type} (s : State α) (d : TileDType) (sh : TileShape) (n : RegName) : Prop :=
  ∃ v, s.regs d sh n = some v

def Holds {α : Type} (c : Shape) (s : State α) : Prop :=
  Present s .nat [] "i_bh" ∧ Present s .nat [] "i_t" ∧
  Present s .nat [c.BT] "rows" ∧ Present s .nat [c.BV] "cols" ∧
  Present s .bool [c.BT,c.BT] "m_s" ∧ Present s .real [c.BT,c.BV] "b_o" ∧
  Present s .real [c.BT,c.BT] "b_s"

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.Typed
