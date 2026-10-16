import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeBits

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeGroup
open NativeDomain NativeBits
set_option maxHeartbeats 3000000

structure Values where
  group : BitVec 32
  first : BitVec 32
  remaining : BitVec 32
  size : BitVec 32
  rowRemainder : BitVec 32
  row : BitVec 32
  columnRemainder : BitVec 32
  column : BitVec 32

/-- The exact source group expression DAG. In particular both remainders use
the absolute tile ID, including the shortened final row group. -/
def eval (p : Parameters) (nr span tile : BitVec 32) : Values :=
  let group := tile.sdiv span
  let first := group * bits (p.GM : Int)
  let remaining := nr - first
  let size := smin remaining (bits (p.GM : Int))
  let rowRemainder := tile.srem size
  let columnRemainder := tile.srem span
  ⟨group,first,remaining,size,rowRemainder,first+rowRemainder,
    columnRemainder,columnRemainder.sdiv size⟩

structure Agrees (p : Parameters) (tile : Nat) (v : Values) : Prop where
  group : Exact v.group (tile/(p.GM*columns p))
  first : Exact v.first (Schedule.groupBase (rows p) (columns p) p.GM tile)
  remaining : Exact v.remaining ((rows p-Schedule.groupBase (rows p) (columns p) p.GM tile : Nat) : Int)
  size : Exact v.size (Schedule.groupSize (rows p) (columns p) p.GM tile)
  rowRemainder : Exact v.rowRemainder (tile%Schedule.groupSize (rows p) (columns p) p.GM tile)
  row : Exact v.row (Schedule.row (rows p) (columns p) p.GM tile)
  columnRemainder : Exact v.columnRemainder (tile%(p.GM*columns p))
  column : Exact v.column (Schedule.column (rows p) (columns p) p.GM tile)

theorem agrees (p : Parameters) (h : Admissible p) (tile : Nat) (ht : tile < total p)
    (nr span tileBits : BitVec 32) (hnr : Exact nr (rows p))
    (hspan : Exact span (p.GM*columns p)) (htile : Exact tileBits tile) :
    Agrees p tile (eval p nr span tileBits) := by
  obtain ⟨_hspan,hq,hf,hspos,_hsle,hr,hc⟩ := group_bounds p h tile ht
  have htbound := lt_trans ht h.total_bound
  have hgm := nat_exact (h.scalar p.GM (by simp))
  have hgroup := div_exact htile hspan htbound
  have hfirst := nat_mul_exact hgroup hgm (lt_trans hf (rows_lt p h))
  have hremaining := nat_sub_exact hnr hfirst hf.le (rows_lt p h)
  have hsize : Exact (smin (nr-(tileBits.sdiv span)*bits (p.GM : Int)) (bits (p.GM : Int)))
      (Schedule.groupSize (rows p) (columns p) p.GM tile) := by
    simpa only [Schedule.groupSize,Nat.cast_min] using min_exact hremaining hgm
  have hrowrem := rem_exact htile hsize
  have hrow := nat_add_exact hfirst hrowrem (lt_trans hr (rows_lt p h))
  have hcolrem := rem_exact htile hspan
  have hcol := div_exact hcolrem hsize
    (lt_of_le_of_lt (Nat.mod_le _ _) htbound)
  exact ⟨hgroup,hfirst,hremaining,hsize,hrowrem,hrow,hcolrem,hcol⟩

/-- Both actual div/rem denominators are positive on every scheduled tile. -/
theorem denominators_positive (p : Parameters) (h : Admissible p) (tile : Nat)
    (ht : tile < total p) :
    0 < p.GM*columns p ∧ 0 < Schedule.groupSize (rows p) (columns p) p.GM tile :=
  ⟨(group_bounds p h tile ht).1,(group_bounds p h tile ht).2.2.2.1⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeGroup
