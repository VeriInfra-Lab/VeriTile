import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Logical
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.PackedStorage

namespace VeriTile.Bench.Repairs.DecayCumsum.LogicalStorage
open VeriTile.Triton Geometry Grid PackedStorage Logical
noncomputable section
set_option maxHeartbeats 4000000
variable {β : Type}

abbrev Data (β : Type) := RegionName → Nat → Nat → Nat → β

def offset (h : Call) (row time col : Nat) : Nat := row*h.headStride+time*h.width+col

def Snapshot (h : Call) (kind : Kind) (storage : Placement h kind) (heap : Heap β) (data : Data β) : Prop :=
  ∀ r∈Roles.inputs kind, ∀ row, row<h.rows → ∀ t, t<h.T → ∀ k, k<h.DK →
    heap (address storage r (offset h row t k))=data r row t k

def crop (h : Call) (kind : Kind) (storage : Placement h kind) (heap : Heap β) (r : RegionName) (i : Fin (Logical.cells h)) : β :=
  let c := coordinates h i
  heap (address storage r (offset h c.1.val c.2.1.val c.2.2.val))

theorem owner_offset (h : Call) (c : Coordinate h) (o : Owner h c) :
    Geometry.offset h o.work.1 o.work.2 o.step.val o.lane.val=offset h c.1.val c.2.1.val c.2.2.val := by
  unfold Geometry.offset offset
  rw [o.rowEq,o.timeEq,o.columnEq]

theorem owner_read (h : Call) (kind : Kind) (storage : Placement h kind) (heap : Heap β) (data : Data β)
    (snapshot : Snapshot h kind storage heap data) (c : Coordinate h) (o : Owner h c)
    (r : RegionName) (hr : r∈Roles.inputs kind) (n : Nat) (hn : n<o.work.1.span) :
    lift storage o.work.1 heap r (relative h o.work.1 o.work.2 n o.lane.val)=
      data r c.1.val (chunkStart h c.2.1.val+n) c.2.2.val := by
  have hc := owner_chunk h c o
  have timeEq : Geometry.time o.work.1 o.work.2.chunk n=chunkStart h c.2.1.val+n := by
    rw [hc.1]
    rfl
  have inside := time_inside h o.work.1 (schedule_valid h o.work o.member) o.work.2 ⟨n,hn⟩
  unfold lift
  rw [relocated]
  have eq : Geometry.offset h o.work.1 o.work.2 n o.lane.val=
      offset h c.1.val (chunkStart h c.2.1.val+n) c.2.2.val := by
    unfold Geometry.offset offset
    rw [o.rowEq,timeEq,o.columnEq]
  rw [eq]
  exact snapshot r hr c.1.val c.1.isLt _ (timeEq ▸ inside) c.2.2.val c.2.2.isLt

#axiomsClean owner_read
end
end VeriTile.Bench.Repairs.DecayCumsum.LogicalStorage
