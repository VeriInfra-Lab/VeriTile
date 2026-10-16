import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordSource
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
local instance scaledReferenceDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : State Word) := s.pids 0 / (p.groupRows*blocksN p) * p.groupRows
def rowsInGroup (p : Parameters) (s : State Word) := min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : State Word) := firstRow p s + s.pids 0 % rowsInGroup p s
def columnBlock (p : Parameters) (s : State Word) := (s.pids 0 % (p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : State Word) (i : Fin p.blockRows) := (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : State Word) (j : Fin p.blockColumns) := (columnBlock p s*p.blockColumns+j.val)%p.columns
def inputInner (p : Parameters) (_s : State Word) (j : Fin p.blockInner) := j.val
def aOffset (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) :=
  inputRow p s i.1*p.inputRowStride+inputInner p s i.2.1*p.inputInnerStride+step*(p.blockInner*p.inputInnerStride)
def bOffset (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) :=
  inputInner p s i.1*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+step*(p.blockInner*p.weightInnerStride)
def aValue (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) : Int :=
  wrap 8 (if p.even=Bool.true ∨ inputInner p s i.2.1 < p.inner-step*p.blockInner then
    (s.mem p.input (aOffset p s step i)).read .int else 0)
def bValue (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Int :=
  wrap 8 (if p.even=Bool.true ∨ inputInner p s i.1 < p.inner-step*p.blockInner then
    (s.mem p.weight (bOffset p s step i)).read .int else 0)
def accumulation (p : Parameters) (s : State Word) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => wrap 32 (accumulation p s k i + (FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i).toInt)

def outputRow (p : Parameters) (s : State Word) (i : CIndex p) := rowBlock p s*p.blockRows+i.1.val
def outputColumn (p : Parameters) (s : State Word) (i : CIndex p) := columnBlock p s*p.blockColumns+i.2.1.val
/-- The original suffix uses column + N*row; its stride arguments are unused. -/
def outputOffset (p : Parameters) (s : State Word) (i : CIndex p) :=
  outputColumn p s i+p.columns*outputRow p s i
def active (p : Parameters) (s : State Word) (i : CIndex p) : Prop :=
  outputRow p s i < p.rows ∧ outputColumn p s i < p.columns
instance (p : Parameters) (s : State Word) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))
/-- No other=0 is inserted for the original scale load. The execution may
use its typed undefined value on an inactive lane; observed values below
read only the actual initial scale row. -/
def rawScale (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  if active p s i then (s.mem p.scale (outputRow p s i)).read .real
  else s.undef .real p.scale (outputRow p s i)
def scaled (cfg : FP.Config) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  FP.mul cfg (FP.round .fp32 cfg (accumulation p s p.steps i : Rat))
    ((s.mem p.scale (outputRow p s i)).read .real)
def expected (cfg : FP.Config) (p : Parameters) (s : State Word) (i : CIndex p) : Int :=
  (toSigned 32 (scaled cfg p s i)).getD 0
def InputDefined (cfg : FP.Config) (p : Parameters) (s : State Word) : Prop :=
  ∀ i, active p s i → (toSigned 32 (scaled cfg p s i)).isSome
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
