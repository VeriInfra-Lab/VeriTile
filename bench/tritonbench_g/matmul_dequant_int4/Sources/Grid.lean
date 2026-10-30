import bench.tritonbench_g.matmul_dequant_int4.Sources.Bounds
import bench.tritonbench_g.matmul_dequant_int4.Sources.GroupedGrid
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
set_option maxHeartbeats 4000000

/-- A geometric program-id map has no arithmetic inputs or output values.
The two constructors below are the actual standalone and fused source grids. -/
structure Grid (rows columns : Nat) where
  br : Nat
  bc : Nat
  brpos : 0<br
  bcpos : 0<bc
  program : Fin (ceil rows br*ceil columns bc) ≃ Fin (ceil rows br) × Fin (ceil columns bc)

def rectangularGrid (rows columns br bc : Nat) (hr : 0<br) (hc : 0<bc) : Grid rows columns :=
  ⟨br,bc,hr,hc,finProdFinEquiv.symm⟩
noncomputable def fusedGrid (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid) : Grid s.m s.n :=
  ⟨t.bm,t.bn,ht.bmpos,ht.bnpos,
   Equiv.ofBijective (GroupedGrid.mapping (blocksM s t) (blocksN s t) t.gm
    (ceil_positive _ _ hs.npos ht.bnpos) ht.gmpos)
    (GroupedGrid.bijective _ _ _ (ceil_positive _ _ hs.npos ht.bnpos) ht.gmpos)⟩

abbrev Grid.Index {rows columns : Nat} (g : Grid rows columns) :=
  Fin (ceil rows g.br*ceil columns g.bc) × Fin g.br × Fin g.bc
def Grid.work (g : Grid rows columns) (i : g.Index) : Work :=
  ⟨(g.program i.1).1*g.br+i.2.1,(g.program i.1).2*g.bc+i.2.2⟩

theorem Grid.work_injective (g : Grid rows columns) : Function.Injective g.work := by
  intro a b h
  have hr := congrArg Work.row h
  have hc := congrArg Work.col h
  dsimp [Grid.work] at hr hc
  have dr := congrArg (fun r => r/g.br) hr
  have dc := congrArg (fun c => c/g.bc) hc
  simp only [div_coordinate _ _ _ g.brpos a.2.1.isLt,div_coordinate _ _ _ g.brpos b.2.1.isLt] at dr
  simp only [div_coordinate _ _ _ g.bcpos a.2.2.isLt,div_coordinate _ _ _ g.bcpos b.2.2.isLt] at dc
  have hp : a.1=b.1 := g.program.injective (Prod.ext (Fin.ext dr) (Fin.ext dc))
  have hi : a.2.1=b.2.1 := Fin.ext (by rw [hp] at hr;omega)
  have hj : a.2.2=b.2.2 := Fin.ext (by rw [hp] at hc;omega)
  exact Prod.ext hp (Prod.ext hi hj)

theorem Grid.covers (g : Grid rows columns) (w : Work) (h : w.Active rows columns) :
    ∃ i : g.Index,g.work i=w := by
  have hr : w.row/g.br<ceil rows g.br :=
    (Nat.div_lt_iff_lt_mul g.brpos).mpr (lt_of_lt_of_le h.1 (ceil_cover rows g.br g.brpos))
  have hc : w.col/g.bc<ceil columns g.bc :=
    (Nat.div_lt_iff_lt_mul g.bcpos).mpr (lt_of_lt_of_le h.2 (ceil_cover columns g.bc g.bcpos))
  let tile : Fin (ceil rows g.br) × Fin (ceil columns g.bc) := (⟨_,hr⟩,⟨_,hc⟩)
  let index : g.Index := (g.program.symm tile,⟨w.row%g.br,Nat.mod_lt _ g.brpos⟩,⟨w.col%g.bc,Nat.mod_lt _ g.bcpos⟩)
  refine ⟨index,?_⟩
  cases w
  simp [Grid.work,index,tile,Nat.div_add_mod']

noncomputable def Grid.schedule (g : Grid rows columns) : List Work :=
  (((Finset.univ : Finset g.Index).toList).filter
    (fun i => decide ((g.work i).Active rows columns))).map g.work

theorem Grid.schedule_mem (g : Grid rows columns) (w : Work) :
    w∈g.schedule ↔ w.Active rows columns := by
  simp only [Grid.schedule,List.mem_map,List.mem_filter,Finset.mem_toList,Finset.mem_univ,
    true_and,decide_eq_true_eq]
  constructor
  · rintro ⟨i,hi,rfl⟩;exact hi
  · intro hw
    obtain ⟨i,hi⟩ := g.covers w hw
    exact ⟨i,by simpa [hi] using hw,hi⟩
theorem Grid.schedule_nodup (g : Grid rows columns) : g.schedule.Nodup := by
  exact (List.nodup_map_iff g.work_injective).mpr ((Finset.nodup_toList _).filter _)

/-- Arbitrary complete CTA/lane serialization is a permutation of the proved
source grid. There are no output-equality or coverage premises. -/
def Grid.Complete (g : Grid rows columns) (order : List Work) := order.Perm g.schedule
theorem Grid.complete_mem (g : Grid rows columns) (order : List Work) (h : g.Complete order) (w : Work) :
    w∈order ↔ w.Active rows columns := h.mem_iff.trans (g.schedule_mem w)
theorem Grid.complete_nodup (g : Grid rows columns) (order : List Work) (h : g.Complete order) :
    order.Nodup := h.nodup_iff.mpr g.schedule_nodup
theorem Grid.formed (g : Grid rows columns) (i : g.Index) :
    (g.work i).row<ceil rows g.br*g.br ∧ (g.work i).col<ceil columns g.bc*g.bc :=
  ⟨rowMajor_bound _ _ _ _ (g.program i.1).1.isLt i.2.1.isLt,
   rowMajor_bound _ _ _ _ (g.program i.1).2.isLt i.2.2.isLt⟩

theorem rectangular_coordinates (rows columns br bc : Nat) (hr : 0<br) (hc : 0<bc)
    (i : (rectangularGrid rows columns br bc hr hc).Index) :
    ((rectangularGrid rows columns br bc hr hc).work i).row=(i.1.val/(ceil columns bc))*br+i.2.1 ∧
    ((rectangularGrid rows columns br bc hr hc).work i).col=(i.1.val%(ceil columns bc))*bc+i.2.2 := by
  constructor <;> rfl
theorem fused_coordinates (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (i : (fusedGrid s t hs ht).Index) :
    ((fusedGrid s t hs ht).work i).row=GroupedGrid.row (blocksM s t) (blocksN s t) t.gm i.1*t.bm+i.2.1 ∧
    ((fusedGrid s t hs ht).work i).col=GroupedGrid.col (blocksM s t) (blocksN s t) t.gm i.1*t.bn+i.2.2 := by
  constructor <;> rfl

#axiomsClean Grid.work_injective
#axiomsClean Grid.covers
#axiomsClean Grid.schedule_mem
#axiomsClean Grid.schedule_nodup
#axiomsClean Grid.formed
#axiomsClean rectangular_coordinates
#axiomsClean fused_coordinates
end VeriTile.Bench.Repairs.MatmulDequantInt4
