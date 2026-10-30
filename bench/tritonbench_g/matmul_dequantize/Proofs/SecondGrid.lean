import bench.tritonbench_g.matmul_dequantize.Proofs.SecondComplete
import bench.tritonbench_g.matmul_dequantize.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.SecondGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
open FPTypedSecond
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,p.groupRows,p.outputRowStride,p.outputColumnStride⟩

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    outputOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  simp only [outputOffset,Geometry.offset,Geometry.outRow,Geometry.outColumn,
    geometry,rowBlock,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,
    Geometry.blockRow,Geometry.blockColumn,Geometry.gridRows,Geometry.gridColumns,
    GroupedGrid.row,GroupedGrid.col,GroupedGrid.first,GroupedGrid.size,atProgram,Nat.mul_comm,ite_true]

theorem result_preserved {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial current : State α) (hout : Outside p.output initial current) (pid : Nat) (i : CIndex p) :
    outputValue M D p (atProgram current pid) i=outputValue M D p (atProgram initial pid) i := by
  have hA : current.mem p.input=initial.mem p.input := funext (fun o => hout _ o ha)
  have hB : current.mem p.weight=initial.mem p.weight := funext (fun o => hout _ o hb)
  have hS : current.mem p.scale=initial.mem p.scale := funext (fun o => hout _ o hs)
  have hZ : current.mem p.zero=initial.mem p.zero := funext (fun o => hout _ o hz)
  have hAv : ∀ n, aValue M p (atProgram current pid) n=aValue M p (atProgram initial pid) n := by
    intro n
    funext j
    simp [aValue,bValue,difference,packedLoad,innerActive,aOffset,bOffset,scaleOffset,zeroOffset,inputRow,inputColumn,inputInner,rowBlock,columnBlock,firstRow,rowsInGroup,atProgram,hA,hB,hS,hZ]
  have hBv : ∀ n, bValue M p (atProgram current pid) n=bValue M p (atProgram initial pid) n := by
    intro n
    funext j
    simp [aValue,bValue,difference,packedLoad,innerActive,aOffset,bOffset,scaleOffset,zeroOffset,inputRow,inputColumn,inputInner,rowBlock,columnBlock,firstRow,rowsInGroup,atProgram,hA,hB,hS,hZ]
  have hacc : ∀ n, accumulation M D p (atProgram current pid) n=accumulation M D p (atProgram initial pid) n := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [accumulation,ih,hAv,hBv]
  simp only [outputValue,hacc]

def program {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (valid : Geometry.Valid (geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output) : Grid.Program (α:=α) (geometry p) p.output where
  run := fun pid s => FP.Structural.exec (backend M D) (source p) (atProgram s pid)
  value := fun s pid i => .mk .fp16 (outputValue M D p (atProgram s pid) i)
  stable := by
    intro initial current hout pid i
    rw [result_preserved M D p ha hb hs hz initial current hout]
  complete := by
    intro pid hp current
    have hw : ∀ i j, active p (atProgram current pid) i → active p (atProgram current pid) j →
        outputOffset p (atProgram current pid) i=outputOffset p (atProgram current pid) j → i=j := by
      intro i j hi hj he
      apply Geometry.offset_injective (geometry p) valid pid i j hi hj
      simpa only [offset_at] using he
    obtain ⟨final,he,hr,hf⟩ := correctness M D p (atProgram current pid) hw
    refine ⟨final,he,?_,?_⟩
    · intro i hi
      simpa only [offset_at] using hr i hi
    · intro r o h
      apply hf r o
      simpa only [offset_at] using h

theorem whole {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (valid : Geometry.Valid (geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State α) (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (geometry p)))) :
    ∃ final, Grid.run (program M D p valid ha hb hs hz) jobs initial=some final ∧
      (∀ pid, pid<Geometry.count (geometry p) → ∀ i, Geometry.Active (geometry p) pid i →
        final.mem p.output (Geometry.offset (geometry p) pid i)=
          .mk .fp16 (outputValue M D p (atProgram initial pid) i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ pid, pid<Geometry.count (geometry p) → ∀ i,
        Geometry.Active (geometry p) pid i → o≠Geometry.offset (geometry p) pid i) →
        final.mem r o=initial.mem r o) :=
  Grid.whole (geometry p) p.output valid (program M D p valid ha hb hs hz) initial jobs schedule

#axiomsClean result_preserved
#axiomsClean whole
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.SecondGrid
