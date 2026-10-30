import bench.tritonbench_g.matmul_triton1.Proofs.Logical
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,1,p.outputRow,p.outputColumn⟩
abbrev LogicalIndex (p : Parameters) := Geometry.Cell (geometry p)

theorem geometry_coordinates (p : Parameters) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) :
    Geometry.blockRow (geometry p) pid=pid/blocksN p ∧
    Geometry.blockColumn (geometry p) pid=pid%blocksN p := by
  have hpid : pid<blocksM p*blocksN p := hp
  have hn : 0<blocksN p := by
    by_contra h
    have hz : blocksN p=0 := by omega
    rw [hz,Nat.mul_zero] at hpid
    omega
  have hs := (GroupedGrid.facts (blocksM p) (blocksN p) 1 pid hn (by decide) hpid).2.1
  have he : GroupedGrid.size (blocksM p) (blocksN p) 1 pid=1 := by
    have hl := min_le_right (blocksM p-GroupedGrid.first 1 (blocksN p) pid) 1
    change GroupedGrid.size (blocksM p) (blocksN p) 1 pid≤1 at hl
    omega
  change GroupedGrid.row (blocksM p) (blocksN p) 1 pid=pid/blocksN p ∧
    GroupedGrid.col (blocksM p) (blocksN p) 1 pid=pid%blocksN p
  simp [GroupedGrid.row,GroupedGrid.col,GroupedGrid.first,he,Nat.mod_one]

theorem row_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    row p (atProgram s pid) i.1=Geometry.outRow (geometry p) pid i := by
  change pid/blocksN p*p.blockRows+i.1.val=Geometry.blockRow (geometry p) pid*p.blockRows+i.1.val
  rw [(geometry_coordinates p pid hp).1]

theorem column_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    column p (atProgram s pid) i.2.1=Geometry.outColumn (geometry p) pid i := by
  change pid%blocksN p*p.blockColumns+i.2.1.val=Geometry.blockColumn (geometry p) pid*p.blockColumns+i.2.1.val
  rw [(geometry_coordinates p pid hp).2]

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by
  simp only [active,row_at p s pid hp i,column_at p s pid hp i]
  rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    cOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  simp only [cOffset,row_at p s pid hp i,column_at p s pid hp i]
  rfl

def one {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) (s : State α) :=
  FP.Structural.exec (backend M P) (source p) (atProgram s pid)
def grid {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :=
  GridExecution.sequence (one M P p) (List.range (Geometry.count (geometry p))) s

theorem single_logical {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial current : State α) (hout : Outside p.output initial current) (pid : Nat) (hp : pid<Geometry.count (geometry p)) :
    ∃ final, one M P p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .real (logical P p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → cOffset p s i=cOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j ((active_at p current pid hp i).mp hi) ((active_at p current pid hp j).mp hj)
    simpa only [s,offset_at p current pid hp] using he
  obtain ⟨final,he,hread,hframe⟩ := correctness M P p s hBK hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := (active_at p current pid hp i).mpr (Geometry.owns_active (geometry p) pid x i hr hc)
    have ho : cOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at p current pid hp]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,accumulation_logical P p s i hi (steps p)]
    rw [show s=atProgram current pid from rfl,row_at p current pid hp i,column_at p current pid hp i]
    change Cell.mk .real (logical P p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,logical_preserved P p initial s ha hb hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i ((active_at p current pid hp i).mp hi)
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : cOffset p s i=Geometry.address (geometry p) x := by
        exact offset_at p current pid hp i
      rw [ho]
      exact h x hx

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (initial : State α) :
    ∃ final, grid M P p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .real (logical P p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : LogicalIndex p => Cell.mk .real (logical P p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one M P p) initial (Geometry.count (geometry p))
    (fun pid hp current hout => single_logical M P p hgeo hBK ha hb initial current hout pid hp)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean geometry_coordinates
#axiomsClean row_at
#axiomsClean column_at
#axiomsClean active_at
#axiomsClean offset_at
#axiomsClean single_logical
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
