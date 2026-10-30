import bench.tritonbench_g.kv_cache_filling.Proofs.QuantProfile
import bench.tritonbench_g.kv_cache_filling.Proofs.Schedule
import bench.tritonbench_g.fast_layernorm.Proofs.Settings

namespace VeriTile.Bench.Repairs.KvCacheFilling.SharedTarget
open QuantProfile
noncomputable section

theorem block_covers (n : Nat) : n ≤ n.nextPowerOfTwo :=
  VeriTile.Bench.Repairs.FastLayernorm.Settings.nextPower_covers n

/-- Ordered source formula. Both quantized branches preserve the exact scalar
operation order and full padded reduction vectors. -/
def row {α : Type} (a : Ops α) (policy : Policy) (w n : Nat) (x : Nat → α) :
    RowResult α (BitVec 8) :=
  match policy with
  | .copy => ⟨fun i => .fp32 (x i), a.literal 0, a.literal 0⟩
  | .int4 => let q := row4 a w n x; ⟨fun i => .uint8 (q.values i), q.scale, q.zero⟩
  | .int8 => let q := row8 a w n x; ⟨fun i => .uint8 (q.values i), q.scale, q.zero⟩

def value {α : Type} (a : Ops α) (r : Request) (k v : Nat → α) : Address → Cell α (BitVec 8) :=
  rowValue (row a r.policy r.ck r.ck.nextPowerOfTwo k)
    (row a r.policy r.cv r.cv.nextPowerOfTwo v)

end
end VeriTile.Bench.Repairs.KvCacheFilling.SharedTarget
