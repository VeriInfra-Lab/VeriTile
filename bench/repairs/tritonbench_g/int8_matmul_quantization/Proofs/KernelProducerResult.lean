import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelProducer

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

abbrev Index (p : Parameters) := TileIndex [1,p.width]
def inputOffset (p : Parameters) (s : State α) (i : Index p) :=
  (s.pids 0 % p.rows)*p.inputRowStride+i.2.1.val*p.inputColumnStride
def outputOffset (p : Parameters) (s : State α) (i : Index p) :=
  (s.pids 0 % p.rows)*p.outputRowStride+i.2.1.val*p.outputColumnStride

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) : α :=
  M.fp32Load (if i.2.1.val < p.columns then (s.mem p.input (inputOffset p s i)).read .real
    else M.literal (some .fp32) .real 0)
def maximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .max (M.literal (some .fp32) .real 0)
    (M.reduceMax (some .fp32) (shape := [1,p.width]) ⟨1, by simp⟩ Bool.false
      (fun i => M.unary (some .fp32) (.abs .real) (loaded M p s i)) (0,PUnit.unit))
def scale {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div (maximum M p s) (M.literal (some .fp32) .real 127)
def quotient {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) : α :=
  M.binary (some .fp32) .real .div (loaded M p s i) (scale M p s)
def rowWritten {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [1,p.width]).foldl (fun t i => if i.2.1.val < p.columns then
    t.write p.output (outputOffset p s i) (.mk .int (integer8 (quotient M p s i)).toInt)
    else t) s
def reference {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) : State α :=
  (rowWritten M integer8 p s).write p.scales (s.pids 0*p.scaleStride)
    (.mk .fp16 (M.cast none .real .fp16 (scale M p s)))

private theorem one_index : TileShape.allIndices [1] = [(0,PUnit.unit)] := by rfl
private theorem fin_one (i : Fin 1) : i = 0 := Subsingleton.elim _ _

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

private theorem fold_expanded {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then
      { t with mem := fun r' o => if r' = r ∧ o = off i then value i else t.mem r' o }
      else t) s).mem =
    l.foldl (fun mem i => fun r' o => if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem :=
  fold_memory l r off value active s

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) (hw : 0 < p.width) :
    (execution M integer8 p s hw).val.mem = (reference M integer8 p s).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [execution, reference,
    rowWritten, one_index, fin_one, fold_memory, fold_expanded, State.write, outputOffset, inputOffset, quotient,
    scale, maximum, loaded, backend, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex,
    show (0.0 : ℝ) = 0 by norm_num, show (127.0 : ℝ) = 127 by norm_num]
  rfl

#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
