import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardState
import VeriTile.Triton.Memory.TraceProofs

/-! Active-address bounds for the exact masked loads and stores. Inactive tile
lanes need no allocation; source loads retain their explicit other=0. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.SafetyAccess
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 700000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) (bounds : RegionBounds) : Prop where
  query : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p l k.val →
    keyAddress p l t k.val < bounds r.q
  key : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p l k.val →
    keyAddress p l t k.val < bounds r.k
  value : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p l j.val →
    valueAddress p l t j.val < bounds r.v
  output : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p l j.val →
    outputAddress p l t j.val < bounds r.o
  seed : hasSeed = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
    activeKey p l k.val → activeValue p l j.val → stateAddress p l k.val j.val < bounds r.initial
  finalState : storeFinal = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
    activeKey p l k.val → activeValue p l j.val → stateAddress p l k.val j.val < bounds r.final

theorem masked_load {shape : TileShape} (bounds : RegionBounds)
    (ptr mask : RegName) (s : BlockState) (addresses : Tile .ptr shape) (active : Tile .bool shape)
    (hp : s.regs .ptr shape ptr = some addresses) (hm : s.regs .bool shape mask = some active)
    (hb : ∀ i, active.data i = Bool.true → (addresses.data i).2 < bounds (addresses.data i).1) :
    (Op.load .real (.ptr (.ref .ptr shape ptr))
      (.maskOther (.ref .bool shape mask) (.broadcast (.const 0) shape))).SafeAt bounds s := by
  simpa [Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, evalOp.eq_def, hp, hm] using hb

theorem masked_store {shape : TileShape} (bounds : RegionBounds)
    (ptr value mask : RegName) (s : BlockState) (addresses : Tile .ptr shape) (active : Tile .bool shape)
    (hp : s.regs .ptr shape ptr = some addresses) (hm : s.regs .bool shape mask = some active)
    (hb : ∀ i, active.data i = Bool.true → (addresses.data i).2 < bounds (addresses.data i).1) :
    Stmt.TraceSafe bounds (.store .real shape (.ptr (.ref .ptr shape ptr))
      (.ref .real shape value) (.mask (.ref .bool shape mask))) s := by
  simpa [Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, evalOp.eq_def, hp, hm] using hb

#axiomsClean masked_load
#axiomsClean masked_store
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.SafetyAccess
