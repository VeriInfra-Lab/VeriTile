import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Mathematical

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FiniteFormula
noncomputable section
set_option maxHeartbeats 1000000

def input (ops : Ops I F H) (s : Shape) (buffers : Buffers) (m : Memory F H) : Mathematical.Inputs s F where
  a := fun r k => read8 m (buffers.a,buffers.aOffset+r.val*s.K+k.val)
  b := fun k n => read8 m (buffers.b,buffers.bOffset+k.val*s.N+n.val)
  rowScale := fun r => read32 ops m (buffers.x,buffers.xOffset+r.val)
  columnScale := fun n => read32 ops m (buffers.w,buffers.wOffset+n.val)
  bias := fun n => read32 ops m (buffers.bias,buffers.biasOffset+n.val)

theorem left (ops : Ops I F H) (s : Shape) (c : Config) (buffers : Buffers) (m : Memory F H)
    (r : Fin s.M) (t z : Nat) (lane : Fin c.BK) :
    Mathematical.left s c (input ops s buffers m) r t z lane = operandA s c buffers m r.val t z lane := by
  unfold Mathematical.left operandA
  split <;> rfl

theorem right (ops : Ops I F H) (s : Shape) (c : Config) (buffers : Buffers) (m : Memory F H)
    (n : Fin s.N) (t z : Nat) (lane : Fin c.BK) :
    Mathematical.right s c (input ops s buffers m) n t z lane = operandB s c buffers m n.val t z lane := by
  unfold Mathematical.right operandB
  split <;> rfl

theorem accumulator (ops : Ops I F H) (s : Shape) (c : Config) (buffers : Buffers)
    (selected : Selection s c) (m : Memory F H) (cell : MatrixCell s) (split : Fin c.S) (t : Nat) :
    Mathematical.accumulator ops s c selected (input ops s buffers m) cell split t =
      accumulation ops s c buffers selected m cell.1.val cell.2.val split.val t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [Mathematical.accumulator,accumulation,ih]
    congr 1

theorem split_value (ops : Ops I F H) (s : Shape) (c : Config) (buffers : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (m : Memory F H)
    (cell : MatrixCell s) (split : Fin c.S) :
    Mathematical.splitValue ops s c selected divisor hasBias (input ops s buffers m) cell split =
      partialValue ops s c buffers selected divisor hasBias m cell.1.val cell.2.val split.val := by
  unfold Mathematical.splitValue partialValue
  rw [accumulator]
  rfl

theorem native (profile : FixedISAProfile) (s : Shape) (c : Config) (buffers : Buffers)
    (selected : Selection s c) (hasBias : Bool) (order : Schedule s c) (m : Memory Word32 Word16)
    (cell : MatrixCell s) :
    Mathematical.native profile s c selected hasBias order (input (fpOps profile) s buffers m) cell =
      wordReference (fpOps profile) s c buffers selected Source.runtimeDivisor hasBias order m cell := by
  unfold Mathematical.native wordReference
  congr 1
  funext split
  exact split_value _ _ _ _ _ _ _ _ _ _

theorem real (s : Shape) (buffers : Buffers) (hasBias : Bool) (m : Memory ℝ ℝ)
    (cell : MatrixCell s) :
    Mathematical.real s hasBias (input realOps s buffers m) cell =
      target s (idealInput s buffers hasBias m) cell.1.val cell.2.val := by
  have sum : (List.ofFn (fun k : Fin s.K =>
      (((input realOps s buffers m).a cell.1 k).toInt : ℝ) *
      (((input realOps s buffers m).b k cell.2).toInt : ℝ))).sum =
      rsum (fun k => (idealInput s buffers hasBias m).A cell.1.val k *
        (idealInput s buffers hasBias m).B k cell.2.val) s.K := by
    rw [List.ofFn_eq_map]
    exact finRange_rsum (fun k => (idealInput s buffers hasBias m).A cell.1.val k *
      (idealInput s buffers hasBias m).B k cell.2.val) s.K
  unfold Mathematical.real
  rw [sum]
  cases hasBias <;> rfl

#axiomsClean real
#axiomsClean accumulator
#axiomsClean split_value
#axiomsClean native
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FiniteFormula
