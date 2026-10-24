import bench.repairs.tritonbench_g.int4_matmul.Formula

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution KernelSource
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- A proof-only total extension of the finite operands; its exterior is never
used by a live source load. It is not part of the public mathematical formula. -/
def operands (s : Shape) (o : Ops α) (v : ChannelValues α (IO.inputs s)) : ScalarState α where
  values := fun r i => match r with
    | .a => if h : i<s.m*s.k then v 0 (⟨i,h⟩,()) else o.zero
    | .scale => if h : i<(s.k/s.group)*s.n then v 2 (⟨i,h⟩,()) else o.zero
    | _ => o.zero
  packed := fun r i => match r with
    | .weight => BitVec.ofInt 32 (if h : i<(s.k/8)*s.n then v 1 (⟨i,h⟩,()) else 0)
    | .zero => BitVec.ofInt 32 (if h : i<(s.k/s.group)*(s.n/8) then v 3 (⟨i,h⟩,()) else 0)

theorem formula_real (s : Shape) (v : ChannelValues ℝ (IO.inputs s)) (r c : Nat) :
    mathematical s (operands s realOps v) r c=Formula.mathematical s v r c := rfl

theorem formula_partial (s : Shape) (t : Tile) (o : Ops α)
    (v : ChannelValues α (IO.inputs s)) (w : Work) :
    referencePartial s t o (operands s o v) w=Formula.partialSum s t o v w.row w.col w.split := rfl

/-- Only live finite input cells are related. No condition is imposed on raw
exterior memory, register scratch or existing output values. -/
structure OperandBinding (s : Shape) (o : Ops α) (v : ChannelValues α (IO.inputs s))
    (initial : RegisterState α) : Prop where
  input : ∀ i,i<s.m*s.k → (initial.mem "a" i).read .real=(operands s o v).values .a i
  scale : ∀ i,i<(s.k/s.group)*s.n → (initial.mem "scale" i).read .real=(operands s o v).values .scale i
  weight : ∀ i,i<(s.k/8)*s.n → (initial.mem "weight" i).read .int=((operands s o v).packed .weight i).toInt
  zero : ∀ i,i<(s.k/s.group)*(s.n/8) → (initial.mem "zero" i).read .int=((operands s o v).packed .zero i).toInt

theorem a_operand (x : Raw.Arguments) (hx : Raw.Geometry x) (t : Tile) (o : Ops α)
    (v : ChannelValues α (IO.inputs x.shape)) (s : RegisterState α)
    (bound : OperandBinding x.shape o v s) (k : Nat)
    (i : CIndex (Raw.parameters x t)) (j : Fin t.bk) :
    aValue o (Raw.parameters x t) s k (i.1,j,())=
      operandA x.shape o (operands x.shape o v) (Raw.work x t s i).row (kIndex t (s.pids 1) k j.val) := by
  simp only [aValue,Raw.inner_kIndex,Raw.parameters_inner,Raw.parameters_input,operandA]
  split_ifs with live
  · rw [Raw.aOffset_dense x hx,bound.input]
    · rfl
    · exact rowMajor_bound _ _ _ _ (Nat.mod_lt _ hx.shape.mpos) live
  · rfl

theorem b_operand (x : Raw.Arguments) (hx : Raw.Geometry x) (t : Tile) (ht : t∈Host.sourceTiles)
    (o : Ops α) (v : ChannelValues α (IO.inputs x.shape)) (s : RegisterState α)
    (bound : OperandBinding x.shape o v s) (k : Nat)
    (i : CIndex (Raw.parameters x t)) (j : Fin t.bk) :
    bValue o (Raw.parameters x t) s k (j,i.2.1,())=
      operandB x.shape o (operands x.shape o v) (Raw.work x t s i).col (kIndex t (s.pids 1) k j.val) := by
  have shifts := Source.shift_position t (Host.all_tiles_valid t ht) (s.pids 1) k j.val
  by_cases live : innerIndex (Raw.parameters x t) s k j<x.shape.k
  · have offsets := Raw.packed_offsets x hx t ht s k (j,i.2.1,()) live
    have live' := live
    rw [Raw.inner_kIndex] at live'
    have cap := active_addresses x.shape hx.shape 0 (inputColumn (Raw.parameters x t) s i.2.1)
      (kIndex t (s.pids 1) k j.val) hx.shape.mpos (Nat.mod_lt _ hx.shape.npos) live'
    simp only [bValue,KernelExecution.difference,Raw.parameters_inner,Raw.parameters_weight,
      Raw.parameters_zero,Raw.parameters_scale,live,ite_true,offsets.1,offsets.2.1,offsets.2.2,
      bound.weight _ cap.2.1,bound.zero _ cap.2.2.2.1,bound.scale _ cap.2.2.1,
      operandB,live',VeriTile.Bench.Repairs.Int4Matmul.difference]
    simp only [signedNibble,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble_packed_word,
      nibble,shifts,inputInner]
    rfl
  · have live' := live
    rw [Raw.inner_kIndex] at live'
    simp [bValue,KernelExecution.difference,live,operandB,live',signedNibble,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble]

private theorem fold_eq (f g : β → γ → β) (xs : List γ)
    (h : ∀ b i,i∈xs → f b i=g b i) (z : β) : xs.foldl f z=xs.foldl g z := by
  induction xs generalizing z with
  | nil => rfl
  | cons a xs ih =>
    simp only [List.foldl_cons]
    rw [h z a (by simp)]
    exact ih (fun b i hi => h b i (by simp [hi])) _

theorem dot_block (x : Raw.Arguments) (hx : Raw.Geometry x) (t : Tile) (ht : t∈Host.sourceTiles)
    (o : Ops α) (v : ChannelValues α (IO.inputs x.shape)) (s : RegisterState α)
    (bound : OperandBinding x.shape o v s) (k : Nat) (seed : CIndex (Raw.parameters x t) → α)
    (i : CIndex (Raw.parameters x t)) :
    dotTile o (aValue o (Raw.parameters x t) s k) (bValue o (Raw.parameters x t) s k) seed i=
      blockStep x.shape t o (operands x.shape o v) (Raw.work x t s i) k (seed i) := by
  unfold dotTile blockStep
  apply fold_eq
  intro acc chunk hc
  have hc' : chunk<t.bk/8 := List.mem_range.mp hc
  have inside (j : Fin 8) : chunk*8+j.val<t.bk := by
    have hb := Nat.div_mul_le_self t.bk 8
    have hj := j.isLt
    omega
  congr 1
  · funext j
    change (if h : chunk*8+j.val<t.bk then
      aValue o (Raw.parameters x t) s k (i.1,⟨chunk*8+j.val,h⟩,()) else o.zero)=_
    rw [dif_pos (inside j)]
    exact a_operand x hx t o v s bound k i ⟨chunk*8+j.val,inside j⟩
  · funext j
    change (if h : chunk*8+j.val<t.bk then
      bValue o (Raw.parameters x t) s k (⟨chunk*8+j.val,h⟩,i.2.1,()) else o.zero)=_
    rw [dif_pos (inside j)]
    exact b_operand x hx t ht o v s bound k i ⟨chunk*8+j.val,inside j⟩

theorem accumulation_blocks (x : Raw.Arguments) (hx : Raw.Geometry x) (t : Tile) (ht : t∈Host.sourceTiles)
    (o : Ops α) (v : ChannelValues α (IO.inputs x.shape)) (s : RegisterState α)
    (bound : OperandBinding x.shape o v s) (k : Nat) (i : CIndex (Raw.parameters x t)) :
    accumulation o (Raw.parameters x t) s k i=
      (List.range k).foldl (fun seed b=>blockStep x.shape t o (operands x.shape o v) (Raw.work x t s i) b seed) o.zero := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [accumulation,dot_block x hx t ht o v s bound,ih]
    simp [List.range_succ,List.foldl_append]

theorem source_partial (x : Raw.Arguments) (hx : Raw.Geometry x) (t : Tile) (ht : t∈Host.sourceTiles)
    (o : Ops α) (v : ChannelValues α (IO.inputs x.shape)) (s : RegisterState α)
    (bound : OperandBinding x.shape o v s) (i : CIndex (Raw.parameters x t)) :
    accumulation o (Raw.parameters x t) s (steps (Raw.parameters x t)) i=
      partialValue x.shape t o (operands x.shape o v) (Raw.work x t s i) :=
  accumulation_blocks x hx t ht o v s bound _ i

#axiomsClean source_partial
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
