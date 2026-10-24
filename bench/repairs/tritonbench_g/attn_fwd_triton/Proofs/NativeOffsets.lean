import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.PhysicalCells
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeArithmetic

/-! Actual widened pointer expressions modulo 2^64. Unused padded pointers
and dead final advances may wrap; exactness is required only at active cells. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeOffsets
open VeriTile.Triton Memory PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Signed tensor/view extents and per-role allocation coverage. These are
ordinary allocation facts, not trace or native-safe index assumptions. -/
structure Allocated (p : Parameters) (alloc : PhysicalCells.Role → Allocation) : Prop where
  widths : ∀ r, (alloc r).width=PhysicalCells.width r
  covered : ∀ z h n d r, PhysicalCells.active p z h n d r →
    PhysicalCells.offset p z h n d r < (alloc r).extent

/-- Sufficient per-role ordinary view windows supply allocation coverage. -/
theorem Allocated.of_windows (p : Parameters) (alloc : PhysicalCells.Role → Allocation)
    (bounds : RegionBounds) (hv : ViewBounds.Windows bounds p)
    (hw : ∀ r, (alloc r).width=PhysicalCells.width r)
    (he : ∀ r, bounds (PhysicalCells.region p r) ≤ (alloc r).extent) :
    Allocated p alloc := by
  refine ⟨hw,?_⟩
  intro z h n d r hr
  cases r with
  | query =>
    rcases hr with ⟨hz,hh,hn,hd⟩
    exact lt_of_lt_of_le (by simpa [PhysicalCells.offset,Memory.BaseOffsets.query] using hv.query z hz h hh n hn d hd) (he .query)
  | key =>
    rcases hr with ⟨hz,hh,hn,hd⟩
    exact lt_of_lt_of_le (by simpa [PhysicalCells.offset,Memory.BaseOffsets.key] using hv.key z hz h hh n hn d hd) (he .key)
  | value =>
    rcases hr with ⟨hz,hh,hn,hd⟩
    exact lt_of_lt_of_le (by simpa [PhysicalCells.offset,Memory.BaseOffsets.value] using hv.value z hz h hh n hn d hd) (he .value)
  | output =>
    rcases hr with ⟨hz,hh,hn,hd⟩
    exact lt_of_lt_of_le (by simpa [PhysicalCells.offset,Memory.BaseOffsets.output] using hv.output z hz h hh n hn d hd) (he .output)
  | queryScale =>
    have hb : p.batches*p.heads*((p.context+127)/128) ≤ bounds p.QS := by simpa using hv.queryScale
    exact lt_of_lt_of_le (lt_of_lt_of_le hr.2.2.2 hb) (he .queryScale)
  | keyScale =>
    have hb : p.batches*p.heads*((p.context+63)/64) ≤ bounds p.KS := by simpa using hv.keyScale
    exact lt_of_lt_of_le (lt_of_lt_of_le hr.2.2.2 hb) (he .keyScale)

def machine (p : Parameters) (z h n d : Nat) : PhysicalCells.Role → Nat
  | .query => matrix p.queryBatch p.queryHead p.queryRow p.queryDim z h n d
  | .key => matrix p.keyBatch p.keyHead p.keyRow p.keyDim z h n d
  | .value => matrix p.valueBatch p.valueHead p.valueRow p.valueDim z h n d
  | .output => matrix p.outputBatch p.outputHead p.outputRow p.outputDim z h n d
  | .queryScale | .keyScale => wrap64 n

theorem machine_mod (p : Parameters) (z h n d : Nat) (r : PhysicalCells.Role) :
    machine p z h n d r=wrap64 (PhysicalCells.offset p z h n d r) := by
  cases r <;> simp only [machine,PhysicalCells.offset,matrix_mod,linearAddress,RealProofs.address]

theorem all_roles (p : Parameters) (alloc : PhysicalCells.Role → Allocation) (h : Allocated p alloc)
    (z head n d : Nat) (r : PhysicalCells.Role) (hr : PhysicalCells.active p z head n d r) :
    machine p z head n d r=PhysicalCells.offset p z head n d r ∧
      signed64 (machine p z head n d r)=(PhysicalCells.offset p z head n d r : Int) ∧
      bytePointer (alloc r).base (PhysicalCells.width r) (machine p z head n d r)=
        (alloc r).base+PhysicalCells.offset p z head n d r*PhysicalCells.width r := by
  simpa only [byteAddress,h.widths r] using
    active_address (alloc r) _ _ (h.covered z head n d r hr) (machine_mod p z head n d r)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeOffsets
