import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.DenseSafety

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.AddressBounds
open VeriTile Triton Whole
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- Capacity facts only. These are discharged from the public IO windows. -/
def Capacity (c : Config) (bounds : RegionName → Nat) : Prop :=
  ∀ r, extent c r ≤ bounds r

theorem gate_bound (c : Config) (a : CTA c) (bounds : RegionName → Nat)
    (cap : Capacity c bounds) (t : Nat) (ht : t<c.T) :
    a.head.val*c.T+t<bounds "g" :=
  lt_of_lt_of_le (pack_bound _ _ _ _ a.head.isLt ht) (by simpa [extent] using cap "g")

theorem value_bound (c : Config) (a : CTA c) (bounds : RegionName → Nat)
    (cap : Capacity c bounds) (r : RegionName) (hr : r="v" ∨ r="do")
    (t j : Nat) (ht : t<c.T) (hj : j<c.V) :
    a.head.val*(c.T*c.V)+t*c.V+j<bounds r := by
  have h:=pack_bound _ _ _ _ (pack_bound _ _ _ _ a.head.isLt ht) hj
  have he : c.heads*c.T*c.V≤bounds r := by
    rcases hr with rfl|rfl
    · simpa [extent] using cap "v"
    · simpa [extent] using cap "do"
  exact lt_of_lt_of_le (by simpa [Nat.add_mul,Nat.mul_assoc] using h) he

theorem history_bound (c : Config) (a : CTA c) (bounds : RegionName → Nat)
    (cap : Capacity c bounds) (r : RegionName) (hr : r="h" ∨ r="dh")
    (j e : Nat) (hj : j<c.V) (he : e<c.K) :
    a.head.val*(c.NT*c.K*c.V)+a.time.val*(c.K*c.V)+j+e*c.V<bounds r := by
  have h:=pack_bound _ _ _ _ (pack_bound _ _ _ _ (pack_bound _ _ _ _ a.head.isLt a.time.isLt) he) hj
  have hb : c.heads*c.NT*c.K*c.V≤bounds r := by
    rcases hr with rfl|rfl
    · simpa [extent] using cap "h"
    · simpa [extent] using cap "dh"
  exact lt_of_lt_of_le (by simpa [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using h) hb

theorem query_bound (c : Config) (a : CTA c) (bounds : RegionName → Nat)
    (cap : Capacity c bounds) (r : RegionName) (hr : r="q" ∨ r="k" ∨ r="dq" ∨ r="dk")
    (t e : Nat) (ht : t<c.T) (he : e<c.K) :
    a.head.val*(c.T*c.K)+t*c.K+e<bounds r := by
  have h:=pack_bound _ _ _ _ (pack_bound _ _ _ _ a.head.isLt ht) he
  have hb : c.heads*c.T*c.K≤bounds r := by
    rcases hr with rfl|rfl|rfl|rfl
    · simpa [extent] using cap "q"
    · simpa [extent] using cap "k"
    · simpa [extent] using cap "dq"
    · simpa [extent] using cap "dk"
  exact lt_of_lt_of_le (by simpa [Nat.add_mul,Nat.mul_assoc] using h) hb

theorem output_gate_bound (c : Config) (a : CTA c) (bounds : RegionName → Nat)
    (cap : Capacity c bounds) (t : Nat) (ht : t<c.T) :
    (a.key.val*c.heads+a.head.val)*c.T+t<bounds "dg" :=
  lt_of_lt_of_le (pack_bound _ _ _ _ (pack_bound _ _ _ _ a.key.isLt a.head.isLt) ht)
    (by simpa [extent] using cap "dg")


end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.AddressBounds
