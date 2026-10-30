import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant32
import bench.tritonbench_g.matmul_dequantize.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.DequantGrid32
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (Outside)
open Dequant32
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,1,p.outputRowStride,p.outputColumnStride⟩

/-- This enumerates the actual two-dimensional launch by a complete permutation;
no synthetic CTA is introduced and neither axis is collapsed in source. -/
def atProgram {α : Type} (p : Parameters) (s : State α) (pid : Nat) : State α :=
  {s with pids := fun ax => if ax=0 then Geometry.blockRow (geometry p) pid
    else if ax=1 then Geometry.blockColumn (geometry p) pid else 0}

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : Index p) :
    active p (atProgram p s pid) i ↔ Geometry.Active (geometry p) pid i := by rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : Index p) :
    outputOffset p (atProgram p s pid) i=Geometry.offset (geometry p) pid i := by rfl

theorem result_preserved {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial current : State α) (hout : Outside p.output initial current) (pid : Nat) (i : Index p) :
    value M p (atProgram p current pid) i=value M p (atProgram p initial pid) i := by
  have hB : current.mem p.weight=initial.mem p.weight := funext (fun o => hout _ o hb)
  have hS : current.mem p.scale=initial.mem p.scale := funext (fun o => hout _ o hs)
  have hZ : current.mem p.zero=initial.mem p.zero := funext (fun o => hout _ o hz)
  simp [value,difference,weightOffset,scaleOffset,zeroOffset,row,column,atProgram,hB,hS,hZ]

def program {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (valid : Geometry.Valid (geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output) :
    Grid.Program (α:=α) (geometry p) p.output where
  run := fun pid s => FP.Structural.exec M (source p) (atProgram p s pid)
  value := fun s pid i => .mk .fp32 (value M p (atProgram p s pid) i)
  stable := by
    intro initial current hout pid i
    rw [result_preserved M p hb hs hz initial current hout]
  complete := by
    intro pid hp current
    have hw : ∀ i j, active p (atProgram p current pid) i → active p (atProgram p current pid) j →
        outputOffset p (atProgram p current pid) i=outputOffset p (atProgram p current pid) j → i=j := by
      intro i j hi hj he
      exact Geometry.offset_injective (geometry p) valid pid i j hi hj he
    obtain ⟨final,he,hr,hf⟩ := correctness M p (atProgram p current pid) hw
    exact ⟨final,he,hr,hf⟩

theorem whole {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (valid : Geometry.Valid (geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State α) (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (geometry p)))) :
    ∃ final, Grid.run (program M p valid hb hs hz) jobs initial=some final ∧
      (∀ pid, pid<Geometry.count (geometry p) → ∀ i, Geometry.Active (geometry p) pid i →
        final.mem p.output (Geometry.offset (geometry p) pid i)=
          .mk .fp32 (value M p (atProgram p initial pid) i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ pid, pid<Geometry.count (geometry p) → ∀ i,
        Geometry.Active (geometry p) pid i → o≠Geometry.offset (geometry p) pid i) →
        final.mem r o=initial.mem r o) :=
  Grid.whole (geometry p) p.output valid (program M p valid hb hs hz) initial jobs schedule

#axiomsClean result_preserved
#axiomsClean whole
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.DequantGrid32
