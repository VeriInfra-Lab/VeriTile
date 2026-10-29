import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordSource
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) : Nat := (p.rows + p.blockRows - 1) / p.blockRows
def blocksN (p : Parameters) : Nat := (p.columns + p.blockColumns - 1) / p.blockColumns
def firstRow {α : Type} (p : Parameters) (s : State α) : Nat :=
  s.pids 0 / (p.groupRows * blocksN p) * p.groupRows
def rowsInGroup {α : Type} (p : Parameters) (s : State α) : Nat :=
  min (blocksM p - firstRow p s) p.groupRows
def rowBlock {α : Type} (p : Parameters) (s : State α) : Nat :=
  firstRow p s + ((s.pids 0 % (p.groupRows * blocksN p)) % rowsInGroup p s)
def columnBlock {α : Type} (p : Parameters) (s : State α) : Nat :=
  (s.pids 0 % (p.groupRows * blocksN p)) / rowsInGroup p s
def inputRow {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockRows) : Nat :=
  (rowBlock p s * p.blockRows + i.val) % p.rows
def inputColumn {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockColumns) : Nat :=
  (columnBlock p s * p.blockColumns + i.val) % p.columns
/-- A advances across both loops; B resets at each packed field. -/
def aOffset {α : Type} (p : Parameters) (s : State α) (field step : Nat) (i : AIndex p) : Nat :=
  inputRow p s i.1 * p.inputRowStride + i.2.1.val * p.inputInnerStride +
    (field * p.steps + step) * (p.blockInner * p.inputInnerStride)
def bOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Nat :=
  i.1.val * p.weightInnerStride + inputColumn p s i.2.1 * p.weightColumnStride +
    step * (p.blockInner * p.weightInnerStride)
def aValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (field step : Nat) (i : AIndex p) : Int :=
  wrap 8 (if i.2.1.val < p.inner - (field * p.steps + step) * p.blockInner then
    (s.mem p.input (aOffset p s field step i)).read .int else 0)
def bValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (field step : Nat) (i : BIndex p) : Int :=
  let packed := if i.1.val < p.inner then (s.mem p.weight (bOffset p s step i)).read .nat else 0
  wrap 8 (Int.ofNat ((packed &&& (3 <<< (2 * field))) >>> (2 * field))) - 1
/-- Per-field and per-K-block word recurrences over initial input memory.
The initial register contents and eventual output memory never enter them. -/
def innerValues {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (field : Nat) (initial : CIndex p → Int) : Nat → CIndex p → Int
  | 0 => initial
  | step + 1 => fun i => wrap 32 (innerValues p s field initial step i +
      wordDot (aValue p s field step) (bValue p s field step) i)
def outerValues {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | field + 1 => innerValues p s field (outerValues p s field) p.steps
def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  p.outputRowStride * (rowBlock p s * p.blockRows + i.1.val) +
    p.outputColumnStride * (columnBlock p s * p.blockColumns + i.2.1.val)
def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  rowBlock p s * p.blockRows + i.1.val < p.rows ∧
    columnBlock p s * p.blockColumns + i.2.1.val < p.columns
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- A stored signed input is an int32 word, packed B is an unsigned byte.
The execution theorem itself is stronger; this predicate states its physical
storage profile independently of any computed result. -/
def Inputs {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  (∀ f, f < 4 → ∀ k, k < p.steps → ∀ i : AIndex p,
    ∃ x : BitVec 32, s.mem p.input (aOffset p s f k i) = .mk .int x.toInt) ∧
  (∀ k, k < p.steps → ∀ i : BIndex p,
    ∃ x : BitVec 8, s.mem p.weight (bOffset p s k i) = .mk .nat x.toNat)
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
