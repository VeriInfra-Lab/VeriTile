import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordLogical
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Triton FP.Structural
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,p.groupRows,p.outputRowStride,p.outputColumnStride⟩
abbrev LogicalIndex (p : Parameters) := Geometry.Cell (geometry p)

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    outputOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  change p.outputRowStride*Geometry.outRow (geometry p) pid i+
    p.outputColumnStride*Geometry.outColumn (geometry p) pid i=_
  simp only [Geometry.offset,geometry,Nat.mul_comm]

theorem logical_preserved {α : Type} [Inhabited α] (p : Parameters) (initial current : State α)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hout : Outside p.output initial current) (r c : Nat) :
    logicalDot p current r c=logicalDot p initial r c := by
  unfold logicalDot
  congr 1
  apply congrArg (fun f => LogicalSum.sum f p.inner)
  funext k
  unfold logicalProduct
  rw [hout p.input _ ha,hout p.weight _ hb]

def one {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (pid : Nat) (s : State α) :=
  FP.Structural.exec (backend M) (source p) (atProgram s pid)
def grid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  GridExecution.sequence (one M p) (List.range (Geometry.count (geometry p))) s

theorem single_logical {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial current : State α) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one M p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalDot p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  obtain ⟨final,he,hread,hframe⟩ := correctness M p hBK s hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : outputOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,output_logical p hBK s i hi]
    change Cell.mk .int (logicalDot p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,logical_preserved p initial s ha hb hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i hi
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : outputOffset p s i=Geometry.address (geometry p) x := by
        exact offset_at p current pid i
      rw [ho]
      exact h x hx

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (initial : State α) :
    ∃ final, grid M p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalDot p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : LogicalIndex p => Cell.mk .int (logicalDot p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one M p) initial (Geometry.count (geometry p))
    (fun pid _ current hout => single_logical M p hgeo hBK ha hb initial current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean active_at
#axiomsClean offset_at
#axiomsClean logical_preserved
#axiomsClean single_logical
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
