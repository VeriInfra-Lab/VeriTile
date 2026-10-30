import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeOffsets
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeScale

/-! Bind the two actual helper-pointer recurrences and packed scale
expressions to the allocation exactness theorem. All intermediate arithmetic
is modular; only active loads and stores require exact final addresses. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeSource
open VeriTile.Triton PhysicalCells NativeOffsets
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section

theorem key_active (p : Parameters) (alloc : PhysicalCells.Role → Allocation) (ha : Allocated p alloc)
    (z h lane d lower iterations : Nat)
    (hr : active p z h (lower+iterations*64+lane) d .key) :
    let m := keyValue p.keyBatch p.keyHead p.keyRow p.keyDim z h lane d lower iterations
    m=offset p z h (lower+iterations*64+lane) d .key ∧
      signed64 m=(offset p z h (lower+iterations*64+lane) d .key : Int) ∧
      bytePointer (alloc .key).base 4 m=byteAddress (alloc .key) (offset p z h (lower+iterations*64+lane) d .key) := by
  simpa only [ha.widths,width] using active_address (alloc .key) _ _
    (ha.covered _ _ _ _ .key hr) (keyValue_mod p.keyBatch p.keyHead p.keyRow p.keyDim z h lane d lower iterations)

theorem value_active (p : Parameters) (alloc : PhysicalCells.Role → Allocation) (ha : Allocated p alloc)
    (z h lane d lower iterations : Nat)
    (hr : active p z h (lower+iterations*64+lane) d .value) :
    let m := keyValue p.valueBatch p.valueHead p.valueRow p.valueDim z h lane d lower iterations
    m=offset p z h (lower+iterations*64+lane) d .value ∧
      signed64 m=(offset p z h (lower+iterations*64+lane) d .value : Int) ∧
      bytePointer (alloc .value).base 4 m=byteAddress (alloc .value) (offset p z h (lower+iterations*64+lane) d .value) := by
  simpa only [ha.widths,width] using active_address (alloc .value) _ _
    (ha.covered _ _ _ _ .value hr) (keyValue_mod p.valueBatch p.valueHead p.valueRow p.valueDim z h lane d lower iterations)

theorem query_scale_active (p : Parameters) (alloc : PhysicalCells.Role → Allocation) (ha : Allocated p alloc)
    (z h block : Nat) (hz : z<p.batches) (hh : h<p.heads) (hb : block<(p.context+127)/128) :
    let offset := (z*p.heads+h)*((p.context+127)/128)+block
    let m := packedScale (z*p.heads+h) ((p.context+127)/128) block
    m=offset ∧ signed64 m=(offset : Int) ∧
      bytePointer (alloc .queryScale).base 4 m=byteAddress (alloc .queryScale) offset := by
  have hx := active_address (alloc .queryScale) _ _
    (ha.covered 0 0 _ 0 .queryScale ⟨rfl,rfl,rfl,ViewBounds.packed _ _ _ _ _ _ hz hh hb⟩)
    (packedScale_mod (z*p.heads+h) ((p.context+127)/128) block)
  simpa only [ha.widths,width] using hx

theorem key_scale_active (p : Parameters) (alloc : PhysicalCells.Role → Allocation) (ha : Allocated p alloc)
    (z h block : Nat) (hz : z<p.batches) (hh : h<p.heads) (hb : block<(p.context+63)/64) :
    let offset := (z*p.heads+h)*((p.context+63)/64)+block
    let m := packedScale (z*p.heads+h) ((p.context+63)/64) block
    m=offset ∧ signed64 m=(offset : Int) ∧
      bytePointer (alloc .keyScale).base 4 m=byteAddress (alloc .keyScale) offset := by
  have hx := active_address (alloc .keyScale) _ _
    (ha.covered 0 0 _ 0 .keyScale ⟨rfl,rfl,rfl,ViewBounds.packed _ _ _ _ _ _ hz hh hb⟩)
    (packedScale_mod (z*p.heads+h) ((p.context+63)/64) block)
  simpa only [ha.widths,width] using hx

/-- K-scale helper starts at lower/64 and advances once per 64-key block. -/
theorem key_scale_progression (head count lower iterations : Nat) :
    NativeOffsets.advance (packedScale head count (lower/64)) (wrap64 1) iterations =
      wrap64 (head*count+(lower+iterations*64)/64) := scale_progression head count lower iterations

theorem key_scale_helper_active (p : Parameters) (alloc : PhysicalCells.Role → Allocation)
    (ha : Allocated p alloc) (z h lower iterations : Nat)
    (hz : z<p.batches) (hh : h<p.heads) (hc : lower+iterations*64<p.context) :
    let offset := (z*p.heads+h)*((p.context+63)/64)+(lower+iterations*64)/64
    let m := NativeOffsets.advance (packedScale (z*p.heads+h) ((p.context+63)/64) (lower/64)) (wrap64 1) iterations
    m=offset ∧ signed64 m=(offset : Int) ∧
      bytePointer (alloc .keyScale).base 4 m=byteAddress (alloc .keyScale) offset := by
  have hb := RealProofs.scale_tail_bound p.context (lower+iterations*64) hc
  have hx := active_address (alloc .keyScale) _ _
    (ha.covered 0 0 _ 0 .keyScale ⟨rfl,rfl,rfl,ViewBounds.packed _ _ _ _ _ _ hz hh hb⟩)
    (key_scale_progression (z*p.heads+h) ((p.context+63)/64) lower iterations)
  simpa only [ha.widths,width] using hx

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeSource
