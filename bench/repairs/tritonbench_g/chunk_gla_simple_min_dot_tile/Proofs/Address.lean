import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Kernels
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.Layout

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Address
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host

/-- All six ordinary-pointer expressions use the original logical addresses.
The memory instructions change representation and choose explicit zero tails;
the dot operands, score cast, gates and scale are unchanged. -/
theorem query_address (c : Shape) (l : Launch) (kb i e : Nat) :
    l.head*(c.T*c.K)+(l.chunk*c.BT+i)*c.K+(kb*c.BK+e) =
      keyAddress (layout c) l (row (layout c) l i) (kb*c.BK+e) := rfl

theorem key_address (c : Shape) (l : Launch) (kb e j : Nat) :
    l.head*(c.T*c.K)+(kb*c.BK+e)+(l.chunk*c.BT+j)*c.K =
      keyAddress (layout c) l (row (layout c) l j) (kb*c.BK+e) := by
  simp [keyAddress,layout,row]
  omega

theorem history_address (c : Shape) (l : Launch) (kb e j : Nat) :
    l.head*(c.NT*c.K*c.V)+l.chunk*c.K*c.V+(kb*c.BK+e)*c.V+(l.value*c.BV+j) =
      historyAddress (layout c) l (kb*c.BK+e) j := rfl

theorem value_output_address (c : Shape) (l : Launch) (i j : Nat) :
    l.head*(c.T*c.V)+(l.chunk*c.BT+i)*c.V+(l.value*c.BV+j) =
      outputAddress (layout c) l i j := rfl

theorem gate_address (c : Shape) (l : Launch) (i : Nat) :
    l.head*c.T+(l.chunk*c.BT+i) = gateAddress (layout c) l i := rfl

/-- Exact source-active read/write bounds, reused for every grid program. -/
theorem accesses_bounded (c : Shape) (idx : GridIndex (grid c)) (A : FlatAlloc)
    (hc : ∀ r, capacity c r ≤ A.extent r) :
    Safe.Bounds (layout c) (launchIndex c idx) regions A.extent :=
  bounds c idx A hc

/-- The candidate output mask is exactly the retained active-output predicate. -/
theorem output_mask (c : Shape) (l : Launch) (i j : Nat) :
    ((l.chunk*c.BT+i < c.T) ∧ (l.value*c.BV+j < c.V)) ↔
      active (layout c) l i j := Iff.rfl

/-- Fresh output coverage and pairwise write disjointness reuse all original
host geometry, independently of input storage aliases before materialization. -/
theorem unique_writers (c : Shape) (l m : Launch) (i j : TileIndex [c.BT,c.BV])
    (hi : active (layout c) l i.1.val i.2.1.val)
    (hj : active (layout c) m j.1.val j.2.1.val)
    (he : outputAddress (layout c) l i.1.val i.2.1.val =
      outputAddress (layout c) m j.1.val j.2.1.val) : l = m ∧ i = j :=
  address_injective c l m i j hi hj he

/-- Native address representability follows from the ordinary allocation
capacity, instead of imposing a smaller shape/tile fixture. Native.lean gives
the complete active-address and padded-mask i64 refinement. -/
theorem signed64_of_capacity (off extent : Nat) (ho : off < extent)
    (he : extent ≤ 2^63) : off < 2^63 := lt_of_lt_of_le ho he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Address
