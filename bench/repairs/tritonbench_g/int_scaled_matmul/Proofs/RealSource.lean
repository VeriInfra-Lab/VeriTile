import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.SourcePreservation
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.ScaledWordGrid

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
abbrev Parameters := ScaledWord.Parameters
abbrev prelude := ScaledWord.prelude
abbrev body := ScaledWord.body
abbrev originalBody := ScaledWord.originalBody
abbrev original := ScaledWord.original
abbrev typedScaled := ScaledWord.typedScaled
abbrev widths := ScaledWord.widths
abbrev widthAssignment := ScaledWord.widthAssignment
abbrev epilogue := ScaledWord.epilogue
abbrev source := Kernels.scaled

def wrap (w : Nat) (x : Int) := (BitVec.ofInt w x).toInt
/-- Independent ideal-Real toward-zero conversion, total at every Real value.
The mathematical signed32 projection follows the retained integer output type.
It does not assert agreement with undefined native floating conversions. -/
def truncate (x : ℝ) : Int := if x<0 then Int.ceil x else Int.floor x

def backend (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    fromNat := fun _ n => (n : ℝ)
    fromInt := fun _ z => (z : ℝ)
    cast := fun _ _ _ x => x
    fp32Load := id
    binary := fun p d op x y => match op with
      | .mul => x*y
      | _ => rest.binary p d op x y
    floatToInt := fun _ _ w => some (fun x => some (wrap w (truncate x)))
    integerDot := fun _ {batch} {_} {_} {_} a b =>
      some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt) }
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : State ℝ) := s.pids 0 / (p.groupRows*blocksN p) * p.groupRows
def rowsInGroup (p : Parameters) (s : State ℝ) := min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : State ℝ) := firstRow p s + s.pids 0 % rowsInGroup p s
def columnBlock (p : Parameters) (s : State ℝ) := (s.pids 0 % (p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : State ℝ) (i : Fin p.blockRows) := (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : State ℝ) (j : Fin p.blockColumns) := (columnBlock p s*p.blockColumns+j.val)%p.columns
def inputInner (p : Parameters) (_s : State ℝ) (j : Fin p.blockInner) := j.val
def aOffset (p : Parameters) (s : State ℝ) (step : Nat) (i : AIndex p) :=
  inputRow p s i.1*p.inputRowStride+inputInner p s i.2.1*p.inputInnerStride+step*(p.blockInner*p.inputInnerStride)
def bOffset (p : Parameters) (s : State ℝ) (step : Nat) (i : BIndex p) :=
  inputInner p s i.1*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+step*(p.blockInner*p.weightInnerStride)
def aValue (p : Parameters) (s : State ℝ) (step : Nat) (i : AIndex p) : Int :=
  wrap 8 (if p.even=Bool.true ∨ inputInner p s i.2.1 < p.inner-step*p.blockInner then
    (s.mem p.input (aOffset p s step i)).read .int else 0)
def bValue (p : Parameters) (s : State ℝ) (step : Nat) (i : BIndex p) : Int :=
  wrap 8 (if p.even=Bool.true ∨ inputInner p s i.1 < p.inner-step*p.blockInner then
    (s.mem p.weight (bOffset p s step i)).read .int else 0)
def accumulation (p : Parameters) (s : State ℝ) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => wrap 32 (accumulation p s k i + (FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i).toInt)

def outputRow (p : Parameters) (s : State ℝ) (i : CIndex p) := rowBlock p s*p.blockRows+i.1.val
def outputColumn (p : Parameters) (s : State ℝ) (i : CIndex p) := columnBlock p s*p.blockColumns+i.2.1.val
/-- The repaired suffix uses the actual output and scale row strides. -/
def outputOffset (p : Parameters) (s : State ℝ) (i : CIndex p) :=
  outputColumn p s i*p.outputColumnStride+outputRow p s i*p.outputRowStride
def active (p : Parameters) (s : State ℝ) (i : CIndex p) : Prop :=
  outputRow p s i < p.rows ∧ outputColumn p s i < p.columns
instance (p : Parameters) (s : State ℝ) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))
/-- No other=0 is inserted for the original scale load. The execution may
use its typed undefined value on an inactive lane; observed values below
read only the actual initial scale row. -/
def rawScale (p : Parameters) (s : State ℝ) (i : CIndex p) : ℝ :=
  if active p s i then (s.mem p.scale (outputRow p s i*p.scaleRowStride)).read .real
  else s.undef .real p.scale (outputRow p s i*p.scaleRowStride)
def scaled (p : Parameters) (s : State ℝ) (i : CIndex p) : ℝ :=
  (accumulation p s p.steps i : ℝ) *
    (s.mem p.scale (outputRow p s i*p.scaleRowStride)).read .real
def expected (p : Parameters) (s : State ℝ) (i : CIndex p) : Int :=
  wrap 32 (truncate (scaled p s i))
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
