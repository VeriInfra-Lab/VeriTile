import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Logical
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GridExecution
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,p.groupRows,p.outputRow,p.outputColumn⟩
abbrev LogicalIndex (p : Parameters) := Geometry.Cell (geometry p)

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    cOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  rfl

def one {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) (s : State α) :=
  FP.Structural.exec (backend M P) (source p) (atProgram s pid)
def grid {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :=
  GridExecution.sequence (one M P p) (List.range (Geometry.count (geometry p))) s

theorem single_logical {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial current : State α) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one M P p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .real (logical P p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → cOffset p s i=cOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  obtain ⟨final,he,hread,hframe⟩ := correctness M P p s hBK hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : cOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,accumulation_logical P p s i hi (steps p)]
    change Cell.mk .real (logical P p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,logical_preserved P p initial s ha hb hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i hi
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : cOffset p s i=Geometry.address (geometry p) x := by
        exact offset_at p current pid i
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
    (fun pid _ current hout => single_logical M P p hgeo hBK ha hb initial current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean active_at
#axiomsClean offset_at
#axiomsClean single_logical
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
