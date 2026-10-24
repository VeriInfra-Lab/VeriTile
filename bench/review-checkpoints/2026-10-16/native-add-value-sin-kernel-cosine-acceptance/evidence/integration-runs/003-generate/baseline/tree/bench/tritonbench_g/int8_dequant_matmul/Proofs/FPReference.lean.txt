import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows, p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner, p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows, p.blockColumns]

def blocksM (p : Parameters) : Nat := (p.rows + p.blockRows - 1) / p.blockRows
def blocksN (p : Parameters) : Nat := (p.columns + p.blockColumns - 1) / p.blockColumns

def firstRow {α : Type} (p : Parameters) (s : State α) : Nat :=
  (s.pids 0 / (p.groupRows * blocksN p)) * p.groupRows

def rowsInGroup {α : Type} (p : Parameters) (s : State α) : Nat :=
  min (blocksM p - firstRow p s) p.groupRows

def rowBlock {α : Type} (p : Parameters) (s : State α) : Nat :=
  firstRow p s + s.pids 0 % rowsInGroup p s

def columnBlock {α : Type} (p : Parameters) (s : State α) : Nat :=
  (s.pids 0 % (p.groupRows * blocksN p)) / rowsInGroup p s

def inputRow {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockRows) : Nat :=
  (rowBlock p s * p.blockRows + i.val) % p.rows

def inputColumn {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockColumns) : Nat :=
  (columnBlock p s * p.blockColumns + j.val) % p.columns

def inputInner {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockInner) : Nat :=
  s.pids 1 * p.blockInner + j.val

/-- Pointer increments follow the actual source, without replacing integer
quotients by a mathematically more convenient divisible-block formula. -/
def aOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : AIndex p) : Nat :=
  inputRow p s i.1 * p.inputRowStride + inputInner p s i.2.1 * p.inputInnerStride +
    step * (p.blockInner * p.inputInnerStride)

def bOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Nat :=
  inputInner p s i.1 * p.weightInnerStride + inputColumn p s i.2.1 * p.weightColumnStride +
    step * (p.blockInner * p.weightInnerStride)

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  p.outputRowStride * (rowBlock p s * p.blockRows + i.1.val) +
    p.outputColumnStride * (columnBlock p s * p.blockColumns + i.2.1.val)

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  rowBlock p s * p.blockRows + i.1.val < p.rows ∧
    columnBlock p s * p.blockColumns + i.2.1.val < p.columns
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))


def aValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (step : Nat) (i : AIndex p) : Int := wrap 8 ((s.mem p.input (aOffset p s step i)).read .int)
def bValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (step : Nat) (i : BIndex p) : Int := wrap 8 ((s.mem p.weight (bOffset p s step i)).read .int)
def accumulation {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | step + 1 => fun i => wrap 32 (accumulation p s step i +
      (FP.IntegerDot.wordDot [] (aValue p s step) (bValue p s step) i).toInt)
def rowFactor {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Fin p.blockRows) : α := M.cast (some .fp32) .fp32 .real
      ((s.mem p.rowScale (inputRow p s i)).read .fp32)
def columnFactor {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Fin p.blockColumns) : α := M.cast (some .fp32) .fp32 .real
      ((s.mem p.columnScale (inputColumn p s i)).read .fp32)
/-- All three multiplications, the final half cast, and the optional half
addition follow source order. There is no reassociation or fused multiply-add. -/
def outputValue {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (divisor : α) (i : CIndex p) : α :=
  let scaled := M.binary (some .fp32) .real .mul (columnFactor M p s i.2.1)
    (M.binary (some .fp32) .real .mul (rowFactor M p s i.1)
      (M.binary (some .fp32) .real .mul (M.fromInt (some .fp32) (accumulation p s p.steps i))
        (M.cast (some .fp32) .fp32 .real divisor)))
  let half := M.cast none .real .fp16 scaled
  if p.hasBias then M.binary none .fp16 .add half
    (M.cast none .fp16 .fp16
      ((s.mem p.bias (columnBlock p s * p.blockColumns + i.2.1.val)).read .fp16))
  else half

/-- Actual initial packed-byte storage, independently of computed output. -/
def ByteInputs {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  (∀ k, k < p.steps → ∀ i : AIndex p, ∃ w : BitVec 8,
    s.mem p.input (aOffset p s k i) = .mk .int w.toInt) ∧
  (∀ k, k < p.steps → ∀ i : BIndex p, ∃ w : BitVec 8,
    s.mem p.weight (bOffset p s k i) = .mk .int w.toInt)

theorem byte_wrap (w : BitVec 8) : wrap 8 w.toInt = w.toInt := by
  simp [wrap]
theorem accumulator_range {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (k : Nat) (i : CIndex p) : -2147483648 ≤ accumulation p s k i ∧
      accumulation p s k i < 2147483648 := by
  cases k with
  | zero => simp [accumulation]
  | succ k => exact ⟨BitVec.le_toInt _, BitVec.toInt_lt⟩
#axiomsClean byte_wrap
#axiomsClean accumulator_range
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
