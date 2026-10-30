import bench.repairs.tritonbench_g.bmm_optimized.Proofs.Memory

namespace VeriTile.Bench.Repairs.BmmOptimized
open VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid

theorem group_size_le (gx group gm : Nat) (hg : 0<gm) : groupSize gx group gm≤gm := by
  unfold groupSize
  split
  · exact Nat.le_of_lt (Nat.mod_lt _ hg)
  · exact le_rfl

theorem row_group (px py gx gy gm : Nat) (hg : 0<gm) (notOne : gm≠1)
    (hx : px<gx) (hy : py<gy) :
    VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile px py gx gy gm/gm =
      (px+py*gx)/(gy*gm) := by
  let pid := px+py*gx
  let group := pid/(gy*gm)
  have base := group_base_bound pid gx gy gm (linear_index_bound px py gx gy hx hy)
  have pos := group_size_positive gx group gm hg base
  have bound := lt_of_lt_of_le (Nat.mod_lt (pid%(gy*gm)) pos) (group_size_le gx group gm hg)
  simp only [VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile,notOne,↓reduceIte]
  change (group*gm+pid%(gy*gm)%groupSize gx group gm)/gm=group
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ hg,Nat.div_eq_of_lt bound]
  simp

/-- Injectivity of the actual source swizzle, including the shortened final
group. Both runtime grid dimensions occur in this proof. -/
theorem swizzle_injective_nat (px py qx qy gx gy gm : Nat) (hg : 0<gm)
    (hx : px<gx) (hy : py<gy) (hqx : qx<gx) (hqy : qy<gy)
    (row : VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile px py gx gy gm =
      VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile qx qy gx gy gm)
    (col : VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile px py gx gy gm =
      VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile qx qy gx gy gm) :
    px=qx ∧ py=qy := by
  by_cases one : gm=1
  · simpa [VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile,
      VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile,one] using And.intro row col
  · have geq := congrArg (fun r => r/gm) row
    dsimp only at geq
    rw [row_group px py gx gy gm hg one hx hy,row_group qx qy gx gy gm hg one hqx hqy] at geq
    simp only [VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile,
      VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile,one,↓reduceIte] at row col
    rw [geq] at row col
    have rem : (px+py*gx)%(gy*gm)%groupSize gx ((qx+qy*gx)/(gy*gm)) gm =
        (qx+qy*gx)%(gy*gm)%groupSize gx ((qx+qy*gx)/(gy*gm)) gm := by omega
    have h1 := Nat.mod_add_div ((px+py*gx)%(gy*gm)) (groupSize gx ((qx+qy*gx)/(gy*gm)) gm)
    have h2 := Nat.mod_add_div ((qx+qy*gx)%(gy*gm)) (groupSize gx ((qx+qy*gx)/(gy*gm)) gm)
    rw [rem,col] at h1
    have inner : (px+py*gx)%(gy*gm)=(qx+qy*gx)%(gy*gm) := by omega
    have ha := Nat.mod_add_div (px+py*gx) (gy*gm)
    have hb := Nat.mod_add_div (qx+qy*gx) (gy*gm)
    rw [inner,geq] at ha
    have linear : px+py*gx=qx+qy*gx := by omega
    have gxpos : 0<gx := by omega
    have divide (x y : Nat) (h : x<gx) : (x+y*gx)/gx=y := by
      rw [Nat.add_mul_div_right _ _ gxpos,Nat.div_eq_of_lt h];simp
    have yy := congrArg (fun z => z/gx) linear
    dsimp only at yy
    rw [divide px py hx,divide qx qy hqx] at yy
    rw [yy] at linear
    exact ⟨by omega,yy⟩

def swizzle (gx gy gm : Nat) (hg : 0<gm) (i : Fin gx×Fin gy) : Fin gx×Fin gy :=
  let valid := coordinates_in_grid i.1.val i.2.val gx gy gm hg i.1.isLt i.2.isLt
  (⟨VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile i.1.val i.2.val gx gy gm,valid.1⟩,
   ⟨VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile i.1.val i.2.val gx gy gm,valid.2⟩)
theorem swizzle_injective (gx gy gm : Nat) (hg : 0<gm) :
    Function.Injective (swizzle gx gy gm hg) := by
  intro a b equal
  have rows := congrArg (fun i => i.1.val) equal
  have cols := congrArg (fun i => i.2.val) equal
  obtain ⟨x,y⟩ := swizzle_injective_nat a.1.val a.2.val b.1.val b.2.val gx gy gm hg
    a.1.isLt a.2.isLt b.1.isLt b.2.isLt rows cols
  exact Prod.ext (Fin.ext x) (Fin.ext y)
theorem swizzle_covers (gx gy gm : Nat) (hg : 0<gm) :
    Function.Surjective (swizzle gx gy gm hg) :=
  Finite.surjective_of_injective (swizzle_injective gx gy gm hg)

theorem block_contains (n tile x : Nat) (ht : 0<tile) (hx : x<n) :
    x/tile<blocks n tile := by
  have cover : n≤blocks n tile*tile :=
    VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMath.blocks_cover n tile ht
  apply (Nat.div_lt_iff_lt_mul ht).mpr
  exact lt_of_lt_of_le hx cover

/-- Every logical output cell has an actual source program and lane; batches,
partial row/column tiles, and the grouped final tail are all covered. -/
theorem source_grid_covers (s : Shape) (c : Config) (x : Cell) (active : s.Active x) :
    ∃ pid∈programs s c,∃ lane∈lanes c,cellOf s c pid lane=x := by
  have bounds := config_bounds c
  let target : Fin (s.gx c)×Fin (s.gy c) :=
    (⟨x.row/c.tm,block_contains s.m c.tm x.row bounds.1 active.2.1⟩,
     ⟨x.column/c.tn,block_contains s.n c.tn x.column bounds.2.2.1 active.2.2⟩)
  obtain ⟨pos,hpos⟩ := swizzle_covers (s.gx c) (s.gy c) c.group bounds.2.2.2.2.1 target
  refine ⟨(pos.1.val,pos.2.val,x.batch),?_,(x.row%c.tm,x.column%c.tn),?_,?_⟩
  · simp only [programs,List.mem_product,List.mem_range]
    exact ⟨pos.1.isLt,pos.2.isLt,active.1⟩
  · simp only [lanes,List.mem_product,List.mem_range]
    exact ⟨Nat.mod_lt _ bounds.1,Nat.mod_lt _ bounds.2.2.1⟩
  · have rows := congrArg (fun i => i.1.val) hpos
    have cols := congrArg (fun i => i.2.val) hpos
    dsimp [swizzle,target] at rows cols
    cases x
    simp only [cellOf,rowTile,columnTile,rows,cols,Cell.mk.injEq,true_and]
    constructor
    · simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div _ c.tm
    · simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div _ c.tn

#axiomsClean swizzle_injective_nat
#axiomsClean swizzle_covers
#axiomsClean source_grid_covers
end VeriTile.Bench.Repairs.BmmOptimized
