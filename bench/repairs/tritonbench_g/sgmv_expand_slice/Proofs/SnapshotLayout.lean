import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.GridStride

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SnapshotLayout

/-- A contiguous three-dimensional weight clone, including arbitrary ignored
strides of singleton dimensions and all empty shapes. -/
structure WeightContiguous (adapters columns inner adapterStride columnStride innerStride : Nat) : Prop where
  adapter : 1 < adapters → 0 < columns → 0 < inner → adapterStride = columns*inner
  column : 0 < adapters → 1 < columns → 0 < inner → columnStride = inner
  inner : 0 < adapters → 0 < columns → 1 < inner → innerStride = 1

theorem weight_linear {adapters columns inner adapterStride columnStride innerStride a c k : Nat}
    (layout : WeightContiguous adapters columns inner adapterStride columnStride innerStride)
    (ha : a < adapters) (hc : c < columns) (hk : k < inner) :
    adapterStride*a+k*innerStride+c*columnStride = (a*columns+c)*inner+k := by
  have hca : adapterStride*a = a*(columns*inner) := by
    by_cases h : 1 < adapters
    · rw [layout.adapter h (by omega) (by omega),Nat.mul_comm]
    · have hz : a = 0 := by omega
      simp [hz]
  have hcc : c*columnStride = c*inner := by
    by_cases h : 1 < columns
    · rw [layout.column (by omega) h (by omega)]
    · have hz : c = 0 := by omega
      simp [hz]
  have hck : k*innerStride = k := by
    by_cases h : 1 < inner
    · rw [layout.inner (by omega) (by omega) h,Nat.mul_one]
    · have hz : k = 0 := by omega
      simp [hz]
  rw [hca,hcc,hck]
  simp only [Nat.add_mul,Nat.mul_assoc]
  omega

#axiomsClean weight_linear
end VeriTile.Bench.Repairs.SgmvExpandSlice.SnapshotLayout
