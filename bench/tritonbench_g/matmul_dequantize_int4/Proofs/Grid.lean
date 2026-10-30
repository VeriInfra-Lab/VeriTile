import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Logical
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,p.groupRows,p.outputRowStride,p.outputColumnStride⟩
abbrev Index (p : Parameters) := Geometry.Cell (geometry p)

theorem row_at (p : Parameters) (s : State Word) (pid : Nat) (i : CIndex p) :
    inputRow p (atProgram s pid) i.1=Geometry.outRow (geometry p) pid i := rfl
theorem column_at (p : Parameters) (s : State Word) (pid : Nat) (i : CIndex p) :
    inputColumn p (atProgram s pid) i.2.1=Geometry.outColumn (geometry p) pid i := rfl
theorem active_at (p : Parameters) (s : State Word) (pid : Nat) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := Iff.rfl
theorem offset_at (p : Parameters) (s : State Word) (pid : Nat) (i : CIndex p) :
    outputOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  simp only [outputOffset,Geometry.offset,Geometry.outRow,Geometry.outColumn,geometry]
  rw [Nat.mul_comm p.outputRowStride,Nat.mul_comm p.outputColumnStride]
  rfl

def one (mma : ISA.MMA16) (p : Parameters) (pid : Nat) (s : State Word) :=
  FPFused.exec (FPSoftware.scalar FPCompiled.unused) (ISA.dot mma) p (atProgram s pid)
def grid (mma : ISA.MMA16) (p : Parameters) (s : State Word) :=
  GridExecution.sequence (one mma p) (List.range (Geometry.count (geometry p))) s

theorem single (mma : ISA.MMA16) (p : Parameters) (hlegal : Logical.Legal p)
    (hgeo : Geometry.Valid (geometry p))
    (ha : p.input≠p.output) (hq : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial current : State Word) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one mma p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : Index p => Cell.mk .fp16 (Logical.output mma p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  obtain ⟨final,he,hread,hframe⟩ := ISA.per_program mma p s hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : outputOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [←ho,hread i hi,Logical.output_global mma p hlegal s i hi]
    rw [show s=atProgram current pid from rfl,row_at,column_at,hr,hc]
    rw [Logical.output_preserved mma p initial (atProgram current pid) ha hq hs hz hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i hi
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : outputOffset p s i=Geometry.address (geometry p) x := offset_at p current pid i
      rw [ho]
      exact h x hx

theorem whole_grid (mma : ISA.MMA16) (p : Parameters) (hlegal : Logical.Legal p)
    (hgeo : Geometry.Valid (geometry p))
    (ha : p.input≠p.output) (hq : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State Word) :
    ∃ final, grid mma p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : Index p => Cell.mk .fp16 (Logical.output mma p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : Index p => Cell.mk .fp16 (Logical.output mma p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one mma p) initial (Geometry.count (geometry p))
    (fun pid _ current hout => single mma p hlegal hgeo ha hq hs hz initial current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean offset_at
#axiomsClean single
#axiomsClean whole_grid
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Grid
