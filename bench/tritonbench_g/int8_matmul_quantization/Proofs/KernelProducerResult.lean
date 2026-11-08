import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerObserved

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

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

theorem rowMemory_covered {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) (covers : p.columns≤p.width) :
    QuantizePass.writeIterations M integer8 p s (scale M p s) 0 (tripCount p) s.mem =
      (rowWritten M integer8 p s).mem := by
  rw [tripCount_covered p width covers]
  by_cases empty : p.columns=0
  · simp [empty,QuantizePass.writeIterations,rowWritten]
  · simp only [if_neg empty,QuantizePass.writeIterations]
    simp only [QuantizePass.writeBlock,rowWritten,fold_memory]
    apply congrArg (fun f => (TileShape.allIndices [1,p.width]).foldl f s.mem)
    funext memory i region offset
    rcases i with ⟨row,column,⟨⟩⟩
    by_cases active : column.val<p.columns
    all_goals simp [active,QuantizePass.quotient,quotient,loaded,QuantizePass.inputOffset,
      QuantizePass.outputOffset,inputOffset,outputOffset]

theorem fullMemory_covered {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) (covers : p.columns≤p.width) :
    fullMemory M integer8 p s=(reference M integer8 p s).mem := by
  funext region offset
  simp only [fullMemory,reference,State.write,scale_matches M p s width covers,
    rowMemory_covered M integer8 p s width covers]
  try rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width) (covers : p.columns≤p.width) :
    (execution M integer8 p s width).val.mem=(reference M integer8 p s).mem := by
  exact (execution_full_memory M integer8 p s width).trans (fullMemory_covered M integer8 p s width covers)

#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
