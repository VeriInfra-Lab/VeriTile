import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GroupedGrid
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry
open VeriTile.Triton
set_option maxHeartbeats 4000000

structure Layout where
  (rows columns blockRows blockColumns groupRows rowStride columnStride : Nat)
abbrev Cell (g : Layout) := Fin g.rows × Fin g.columns
abbrev Lane (g : Layout) := TileIndex [g.blockRows,g.blockColumns]
def gridRows (g : Layout) := (g.rows+g.blockRows-1)/g.blockRows
def gridColumns (g : Layout) := (g.columns+g.blockColumns-1)/g.blockColumns
def count (g : Layout) := gridRows g*gridColumns g
def blockRow (g : Layout) (pid : Nat) := GroupedGrid.row (gridRows g) (gridColumns g) g.groupRows pid
def blockColumn (g : Layout) (pid : Nat) := GroupedGrid.col (gridRows g) (gridColumns g) g.groupRows pid
def outRow (g : Layout) (pid : Nat) (i : Lane g) := blockRow g pid*g.blockRows+i.1.val
def outColumn (g : Layout) (pid : Nat) (i : Lane g) := blockColumn g pid*g.blockColumns+i.2.1.val
def address (g : Layout) (x : Cell g) := x.1.val*g.rowStride+x.2.val*g.columnStride
def offset (g : Layout) (pid : Nat) (i : Lane g) := outRow g pid i*g.rowStride+outColumn g pid i*g.columnStride
def Active (g : Layout) (pid : Nat) (i : Lane g) := outRow g pid i<g.rows ∧ outColumn g pid i<g.columns
def index (g : Layout) (pid : Nat) (i : Lane g) (h : Active g pid i) : Cell g :=
  (⟨outRow g pid i,h.1⟩,⟨outColumn g pid i,h.2⟩)
def Owns (g : Layout) (pid : Nat) (x : Cell g) : Prop :=
  ∃ i : Lane g, outRow g pid i=x.1.val ∧ outColumn g pid i=x.2.val
structure Valid (g : Layout) : Prop where
  blockRows : 0<g.blockRows
  blockColumns : 0<g.blockColumns
  groupRows : 0<g.groupRows
  writable : Function.Injective (address g)

theorem lane_injective (g : Layout) (pid : Nat) (i j : Lane g)
    (hr : outRow g pid i=outRow g pid j) (hc : outColumn g pid i=outColumn g pid j) : i=j := by
  apply Prod.ext
  · apply Fin.ext; unfold outRow at hr; omega
  · apply Prod.ext
    · apply Fin.ext; unfold outColumn at hc; omega
    · exact Subsingleton.elim _ _

theorem offset_injective (g : Layout) (h : Valid g) (pid : Nat) (i j : Lane g)
    (hi : Active g pid i) (hj : Active g pid j) (he : offset g pid i=offset g pid j) : i=j := by
  have hx : index g pid i hi=index g pid j hj := h.writable he
  exact lane_injective g pid i j (congrArg (fun x : Cell g => x.1.val) hx)
    (congrArg (fun x : Cell g => x.2.val) hx)

theorem owns_active (g : Layout) (pid : Nat) (x : Cell g) (i : Lane g)
    (hr : outRow g pid i=x.1.val) (hc : outColumn g pid i=x.2.val) : Active g pid i := by
  constructor
  · rw [hr]; exact x.1.isLt
  · rw [hc]; exact x.2.isLt

theorem covers (g : Layout) (h : Valid g) (x : Cell g) :
    ∃ pid, pid<count g ∧ Owns g pid x := by
  have hM := LogicalSum.ceil_cover g.rows g.blockRows h.blockRows
  have hN := LogicalSum.ceil_cover g.columns g.blockColumns h.blockColumns
  have hgn : 0<gridColumns g := by
    by_contra hn
    have hz : gridColumns g=0 := by omega
    change g.columns≤gridColumns g*g.blockColumns at hN
    rw [hz,Nat.zero_mul] at hN
    have hx := x.2.isLt
    omega
  have hr : x.1.val/g.blockRows<gridRows g :=
    (Nat.div_lt_iff_lt_mul h.blockRows).mpr (lt_of_lt_of_le x.1.isLt hM)
  have hc : x.2.val/g.blockColumns<gridColumns g :=
    (Nat.div_lt_iff_lt_mul h.blockColumns).mpr (lt_of_lt_of_le x.2.isLt hN)
  obtain ⟨pid,hpid,hpr,hpc⟩ := GroupedGrid.covers (gridRows g) (gridColumns g) g.groupRows
    hgn h.groupRows _ _ hr hc
  refine ⟨pid,hpid,⟨(⟨x.1.val%g.blockRows,Nat.mod_lt _ h.blockRows⟩,
    ⟨x.2.val%g.blockColumns,Nat.mod_lt _ h.blockColumns⟩,PUnit.unit),?_,?_⟩⟩
  · change blockRow g pid*g.blockRows+x.1.val%g.blockRows=x.1.val
    rw [show blockRow g pid=x.1.val/g.blockRows from hpr]
    exact Nat.div_add_mod' _ _
  · change blockColumn g pid*g.blockColumns+x.2.val%g.blockColumns=x.2.val
    rw [show blockColumn g pid=x.2.val/g.blockColumns from hpc]
    exact Nat.div_add_mod' _ _

theorem owner_unique (g : Layout) (h : Valid g) (p q : Nat) (hp : p<count g) (hq : q<count g)
    (x : Cell g) (hpx : Owns g p x) (hqx : Owns g q x) : p=q := by
  obtain ⟨i,hri,hci⟩ := hpx
  obtain ⟨j,hrj,hcj⟩ := hqx
  have hN := LogicalSum.ceil_cover g.columns g.blockColumns h.blockColumns
  have hgn : 0<gridColumns g := by
    by_contra hn
    have hz : gridColumns g=0 := by omega
    change g.columns≤gridColumns g*g.blockColumns at hN
    rw [hz,Nat.zero_mul] at hN
    have hx := x.2.isLt
    omega
  apply GroupedGrid.injective (gridRows g) (gridColumns g) g.groupRows hgn h.groupRows p q hp hq
  · have hi := Nat.add_mul_div_left i.1.val (blockRow g p) h.blockRows
    have hj := Nat.add_mul_div_left j.1.val (blockRow g q) h.blockRows
    have hd1 : i.1.val/g.blockRows=0 := Nat.div_eq_of_lt i.1.isLt
    have hd2 : j.1.val/g.blockRows=0 := Nat.div_eq_of_lt j.1.isLt
    have he : outRow g p i=outRow g q j := hri.trans hrj.symm
    unfold outRow at he
    rw [Nat.add_comm,Nat.mul_comm] at hi hj
    rw [he] at hi
    change blockRow g p=blockRow g q
    omega
  · have hi := Nat.add_mul_div_left i.2.1.val (blockColumn g p) h.blockColumns
    have hj := Nat.add_mul_div_left j.2.1.val (blockColumn g q) h.blockColumns
    have hd1 : i.2.1.val/g.blockColumns=0 := Nat.div_eq_of_lt i.2.1.isLt
    have hd2 : j.2.1.val/g.blockColumns=0 := Nat.div_eq_of_lt j.2.1.isLt
    have he : outColumn g p i=outColumn g q j := hci.trans hcj.symm
    unfold outColumn at he
    rw [Nat.add_comm,Nat.mul_comm] at hi hj
    rw [he] at hi
    change blockColumn g p=blockColumn g q
    omega

theorem stores_disjoint (g : Layout) (h : Valid g) (p q : Nat)
    (hp : p<count g) (hq : q<count g) (hpq : p≠q)
    (i j : Lane g) (hi : Active g p i) (hj : Active g q j) : offset g p i≠offset g q j := by
  intro he
  have hx : index g p i hi=index g q j hj := h.writable he
  have hr := congrArg (fun x : Cell g => x.1.val) hx
  have hc := congrArg (fun x : Cell g => x.2.val) hx
  apply hpq
  apply owner_unique g h p q hp hq (index g p i hi)
  · exact ⟨i,rfl,rfl⟩
  · exact ⟨j,hr.symm,hc.symm⟩

#axiomsClean lane_injective
#axiomsClean offset_injective
#axiomsClean covers
#axiomsClean owner_unique
#axiomsClean stores_disjoint
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry
