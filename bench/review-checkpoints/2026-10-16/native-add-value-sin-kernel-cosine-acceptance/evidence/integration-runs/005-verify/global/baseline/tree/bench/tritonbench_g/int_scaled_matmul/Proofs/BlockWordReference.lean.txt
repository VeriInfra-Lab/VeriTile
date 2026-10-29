import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordSource
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
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

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  p.outputRowStride * (rowBlock p s * p.blockRows + i.1.val) +
    p.outputColumnStride * (columnBlock p s * p.blockColumns + i.2.1.val)

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  rowBlock p s * p.blockRows + i.1.val < p.rows ∧
    columnBlock p s * p.blockColumns + i.2.1.val < p.columns
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))



def steps (p : Parameters) := (p.inner + p.blockInner - 1) / p.blockInner

def aPointer {α : Type} (p : Parameters) (s : State α) (step : Nat) : BlockPtr :=
  ⟨p.input, 0, [p.rows,p.inner], [p.blockRows,p.blockInner],
    [p.inputRowStride,p.inputInnerStride], [rowBlock p s * p.blockRows,step*p.blockInner]⟩
def bPointer {α : Type} (p : Parameters) (s : State α) (step : Nat) : BlockPtr :=
  ⟨p.weight, 0, [p.inner,p.columns], [p.blockInner,p.blockColumns],
    [p.weightInnerStride,p.weightColumnStride], [step*p.blockInner,columnBlock p s * p.blockColumns]⟩
def cPointer {α : Type} (p : Parameters) (s : State α) : BlockPtr :=
  ⟨p.output, 0, [p.rows,p.columns], [p.blockRows,p.blockColumns],
    [p.outputRowStride,p.outputColumnStride], [rowBlock p s*p.blockRows,columnBlock p s*p.blockColumns]⟩

/-- Initial typed storage or the initial typed byte oracle, at the actual
block-pointer address. The unchecked K tail contributes to the result. -/
def loaded {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape) : Int :=
  let ix := TileShape.indexToList shape i
  wrap 8 (if ptr.inBounds ix [0,1] then (s.mem ptr.region (ptr.address ix)).read .int
    else s.undef .int ptr.region (ptr.address ix))
def aValue {α : Type} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) : AIndex p → Int :=
  loaded s (aPointer p s k)
def bValue {α : Type} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) : BIndex p → Int :=
  loaded s (bPointer p s k)
def accumulation {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => wrap 32 (accumulation p s k i +
      (FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i).toInt)
def outputValue {α : Type} [Inhabited α] (p : Parameters) (s : State α) := accumulation p s (steps p)

theorem before_stop (p : Parameters) (hBK : 0 < p.blockInner) (k : Nat) :
    k < steps p ↔ k * p.blockInner < p.inner := by
  have hq := Nat.div_add_mod (p.inner+p.blockInner-1) p.blockInner
  have hm := Nat.mod_lt (p.inner+p.blockInner-1) hBK
  have hbound : p.inner ≤ steps p * p.blockInner := by
    unfold steps
    rw [Nat.mul_comm p.blockInner] at hq
    omega
  constructor
  · intro hk
    have hmul := Nat.mul_le_mul_right p.blockInner (Nat.succ_le_of_lt hk)
    simp only [Nat.succ_mul] at hmul
    unfold steps at hmul
    rw [Nat.mul_comm p.blockInner] at hq
    omega
  · intro hk
    by_contra hn
    have hmul := Nat.mul_le_mul_right p.blockInner (Nat.le_of_not_gt hn)
    omega

#axiomsClean before_stop
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
