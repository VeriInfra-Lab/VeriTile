import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.OperandBridge

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution.Raw
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- Only scalar kernel stride arguments; no Tensor descriptor or API flags. -/
structure Strides where
  rowStride : Nat
  columnStride : Nat

def Strides.elementOffset (s : Strides) (row column : Nat) :=
  row*s.rowStride+column*s.columnStride

structure Arguments where
  shape : Shape
  input : Strides
  packed : Strides
  scale : Strides
  zeros : Strides

/-- Raw address correspondence to dense logical operands. This includes the
original singleton-axis layouts; it does not require canonical packed strides. -/
structure Geometry (x : Arguments) : Prop where
  shape : x.shape.Valid
  input : x.input.rowStride=x.shape.k ∧ x.input.columnStride=1
  active : ∀ k c,k<x.shape.k → c<x.shape.n →
    x.packed.elementOffset (k/8) c=qAddress x.shape k c ∧
    x.scale.elementOffset (k/x.shape.group) c=sAddress x.shape k c ∧
    x.zeros.elementOffset (k/x.shape.group) (c/8)=zAddress x.shape k c

def parameters (x : Arguments) (t : Tile) : Parameters :=
  ⟨"a","weight","out","scale","zero",x.shape.m,x.shape.n,x.shape.k,
    x.input.rowStride,x.input.columnStride,x.packed.rowStride,x.packed.columnStride,
    x.shape.n,1,x.scale.rowStride,x.scale.columnStride,x.zeros.rowStride,x.zeros.columnStride,
    x.shape.group,t.bm,t.bn,t.bk,t.gm,t.splits⟩

@[simp] theorem parameters_input (x : Arguments) (t : Tile) : (parameters x t).input="a" := rfl
@[simp] theorem parameters_weight (x : Arguments) (t : Tile) : (parameters x t).weight="weight" := rfl
@[simp] theorem parameters_scale (x : Arguments) (t : Tile) : (parameters x t).scale="scale" := rfl
@[simp] theorem parameters_zero (x : Arguments) (t : Tile) : (parameters x t).zero="zero" := rfl
@[simp] theorem parameters_inner (x : Arguments) (t : Tile) : (parameters x t).inner=x.shape.k := rfl

def lane (x : Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (parameters x t)) : Lane :=
  ⟨s.pids 0,s.pids 1,i.1.val,i.2.1.val⟩
def work (x : Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (parameters x t)) : Work :=
  laneWork x.shape t (lane x t s i)

theorem row_work (x : Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (parameters x t)) :
    rowBlock (parameters x t) s*t.bm+i.1.val=(work x t s i).row := rfl
theorem column_work (x : Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (parameters x t)) :
    columnBlock (parameters x t) s*t.bn+i.2.1.val=(work x t s i).col := rfl
theorem split_work (x : Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (parameters x t)) :
    (work x t s i).split=s.pids 1 := rfl
theorem inner_kIndex (x : Arguments) (t : Tile) (s : RegisterState α) (k : Nat) (j : Fin t.bk) :
    innerIndex (parameters x t) s k j=kIndex t (s.pids 1) k j.val := by
  simp [innerIndex,inputInner,parameters,kIndex,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]

theorem aOffset_dense (x : Arguments) (hx : Geometry x) (t : Tile)
    (s : RegisterState α) (k : Nat) (i : AIndex (parameters x t)) :
    aOffset (parameters x t) s k i=
      aAddress x.shape (inputRow (parameters x t) s i.1) (kIndex t (s.pids 1) k i.2.1.val) := by
  simp only [aOffset,parameters,hx.input.1,hx.input.2,Nat.mul_one,aAddress,kIndex,inputInner]
  ring

theorem packed_offsets (x : Arguments) (hx : Geometry x) (t : Tile) (ht : t∈Host.sourceTiles)
    (s : RegisterState α) (k : Nat) (i : BIndex (parameters x t))
    (hk : innerIndex (parameters x t) s k i.1<x.shape.k) :
    bOffset (parameters x t) s k i=qAddress x.shape (kIndex t (s.pids 1) k i.1.val)
        (inputColumn (parameters x t) s i.2.1) ∧
    scaleOffset (parameters x t) s k i=sAddress x.shape (kIndex t (s.pids 1) k i.1.val)
        (inputColumn (parameters x t) s i.2.1) ∧
    zeroOffset (parameters x t) s k i=zAddress x.shape (kIndex t (s.pids 1) k i.1.val)
        (inputColumn (parameters x t) s i.2.1) := by
  have hc : inputColumn (parameters x t) s i.2.1<x.shape.n := Nat.mod_lt _ hx.shape.npos
  rw [inner_kIndex] at hk
  have active := hx.active _ _ hk hc
  refine ⟨?_,?_,?_⟩
  · unfold bOffset
    have advance := packed_advance t (Host.all_tiles_valid t ht) (s.pids 1) k i.1.val x.packed.rowStride
    dsimp only [parameters,inputInner]
    rw [Nat.add_right_comm _ _ (k*_),advance]
    exact active.1
  · simp only [scaleOffset,inner_kIndex]
    exact active.2.1
  · simp only [zeroOffset,inner_kIndex]
    exact active.2.2

theorem aValue_operand [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (o : Ops α) (s : RegisterState α) (m : ScalarState α) (binding : InputBinding s m)
    (k : Nat) (i : CIndex (parameters x t)) (j : Fin t.bk) :
    aValue o (parameters x t) s k (i.1,j,PUnit.unit)=
      operandA x.shape o m (work x t s i).row (kIndex t (s.pids 1) k j.val) := by
  simp only [aValue,inner_kIndex,parameters_inner,parameters_input,operandA]
  split_ifs
  · rw [aOffset_dense x hx, binding.input]
    rfl
  · rfl

theorem bValue_operand [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (s : RegisterState α) (m : ScalarState α)
    (binding : InputBinding s m) (k : Nat) (i : CIndex (parameters x t)) (j : Fin t.bk) :
    bValue o (parameters x t) s k (j,i.2.1,PUnit.unit)=
      operandB x.shape o m (work x t s i).col (kIndex t (s.pids 1) k j.val) := by
  have shifts := Source.shift_position t (Host.all_tiles_valid t ht) (s.pids 1) k j.val
  by_cases hk : innerIndex (parameters x t) s k j<x.shape.k
  · have offsets := packed_offsets x hx t ht s k (j,i.2.1,PUnit.unit) hk
    have hk' := hk
    rw [inner_kIndex] at hk'
    simp only [bValue,difference,parameters_inner,parameters_weight,parameters_zero,parameters_scale,
      hk,ite_true,offsets.1,offsets.2.1,offsets.2.2,
      binding.weight,binding.zero,binding.scale,operandB,hk',ite_true,
      VeriTile.Bench.Repairs.Int4Matmul.difference]
    simp only [signedNibble,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble_packed_word,
      nibble,shifts,inputInner]
    rfl
  · have hk' := hk
    rw [inner_kIndex] at hk'
    simp [bValue,difference,hk,operandB,hk',signedNibble,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble]

private theorem fold_eq (f g : β → γ → β) (xs : List γ)
    (h : ∀ b i,i∈xs → f b i=g b i) (z : β) : xs.foldl f z=xs.foldl g z := by
  induction xs generalizing z with
  | nil => rfl
  | cons a xs ih =>
    simp only [List.foldl_cons]
    rw [h z a (by simp)]
    exact ih (fun b i hi => h b i (by simp [hi])) _

theorem dotTile_block [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (s : RegisterState α) (m : ScalarState α)
    (binding : InputBinding s m) (k : Nat) (seed : CIndex (parameters x t) → α)
    (i : CIndex (parameters x t)) :
    dotTile o (aValue o (parameters x t) s k) (bValue o (parameters x t) s k) seed i=
      blockStep x.shape t o m (work x t s i) k (seed i) := by
  unfold dotTile blockStep
  apply fold_eq
  intro acc chunk hc
  have hc' : chunk<t.bk/8 := List.mem_range.mp hc
  have bound (j : Fin 8) : chunk*8+j.val<t.bk := by
    have hb := Nat.div_mul_le_self t.bk 8
    have hj := j.isLt
    omega
  congr 1
  · funext j
    change (if h : chunk*8+j.val<t.bk then
      aValue o (parameters x t) s k (i.1,⟨chunk*8+j.val,h⟩,PUnit.unit) else o.zero)=_
    rw [dif_pos (bound j)]
    exact aValue_operand x hx t o s m binding k i ⟨chunk*8+j.val,bound j⟩
  · funext j
    change (if h : chunk*8+j.val<t.bk then
      bValue o (parameters x t) s k (⟨chunk*8+j.val,h⟩,i.2.1,PUnit.unit) else o.zero)=_
    rw [dif_pos (bound j)]
    exact bValue_operand x hx t ht o s m binding k i ⟨chunk*8+j.val,bound j⟩

theorem accumulation_blocks [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (s : RegisterState α) (m : ScalarState α)
    (binding : InputBinding s m) (k : Nat) (i : CIndex (parameters x t)) :
    accumulation o (parameters x t) s k i=
      (List.range k).foldl (fun seed b => blockStep x.shape t o m (work x t s i) b seed) o.zero := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [accumulation,dotTile_block x hx t ht o s m binding,ih]
    simp [List.range_succ,List.foldl_append]

/-- Complete source recurrence equals the existing packed-matrix result model.
The connecting obligations contain only scalar strides and input words. -/
theorem source_partial [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (s : RegisterState α) (m : ScalarState α)
    (binding : InputBinding s m) (i : CIndex (parameters x t)) :
    accumulation o (parameters x t) s (steps (parameters x t)) i=
      partialValue x.shape t o m (work x t s i) :=
  accumulation_blocks x hx t ht o s m binding _ i


/-- The unchanged Tensor guard supplies exactly these raw scalar premises. -/
def ofHost (x : Host.Arguments) : Arguments :=
  ⟨x.shape,⟨x.input.rowStride,x.input.columnStride⟩,
    ⟨x.packed.rowStride,x.packed.columnStride⟩,
    ⟨x.scale.rowStride,x.scale.columnStride⟩,
    ⟨x.zeros.rowStride,x.zeros.columnStride⟩⟩

theorem ofHost_geometry (x : Host.Arguments) (hx : Host.Contract x)
    (t : Tile) (ht : t∈Host.sourceTiles) : Geometry (ofHost x) :=
  ⟨hx.shape,⟨hx.input.2.2.1,hx.input.2.2.2.1⟩,
    (Host.contract_strides x hx t ht).active⟩

@[simp] theorem parameters_ofHost (x : Host.Arguments) (t : Tile) :
    parameters (ofHost x) t=KernelExecution.parameters x t := rfl

#axiomsClean source_partial
#axiomsClean ofHost_geometry
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution.Raw
