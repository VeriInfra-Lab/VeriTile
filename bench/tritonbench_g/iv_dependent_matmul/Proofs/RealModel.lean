import bench.tritonbench_g.iv_dependent_matmul.Kernels
import bench.tritonbench_g.iv_dependent_matmul.Sources.Mathematical_Schedules

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.RealExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels
open TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

/-- Mathematical matrix primitive: a finite sum, independently defined over
arbitrary Real operands. Floating casts are identities only in this model. -/
def sumDot {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → ℝ) (b : TileIndex (batch++[k,n]) → ℝ) :
    TileIndex (batch++[m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ j : Fin k, a (i.1,j,PUnit.unit)*b (j,i.2.1,PUnit.unit)
  | _::tail => fun i => sumDot (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def backend (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    binary := fun p d op x y => match op with
      | .add => x+y | _ => rest.binary p d op x y
    cast := fun _ _ _ x => x
    fp32Load := id
    dotTyped := fun _ l r o _ {_} {_} {_} {_} a b => some (fun i =>
      ofFloat o (sumDot (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i))
    dotAcc := fun _ l r o _ {_} {_} {_} {_} a b c => some (fun i =>
      ofFloat o (toFloat o (c i) + sumDot (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i)) }
def columnsGrid (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def rowBlock (p : Parameters) (s : State ℝ) := s.pids 0/columnsGrid p
def colBlock (p : Parameters) (s : State ℝ) := s.pids 0%columnsGrid p
def row (p : Parameters) (s : State ℝ) (i : Fin p.blockRows) := rowBlock p s*p.blockRows+i.val
def col (p : Parameters) (s : State ℝ) (j : Fin p.blockColumns) := colBlock p s*p.blockColumns+j.val

def aBase (p : Parameters) (s : State ℝ) (i : TileIndex [p.blockRows,p.blockInner]) :=
  (row p s i.1%p.rows)*p.inputRowStride+i.2.1.val*p.inputInnerStride
def bBase (p : Parameters) (s : State ℝ) (i : TileIndex [p.blockInner,p.blockColumns]) :=
  i.1.val*p.weightInnerStride+(col p s i.2.1%p.columns)*p.weightColumnStride
def aOffset (p : Parameters) (s : State ℝ) (cur : Nat) (i : TileIndex [p.blockRows,p.blockInner]) :=
  aBase p s i+cur*p.blockInner*p.inputInnerStride
def bOffset (p : Parameters) (s : State ℝ) (cur : Nat) (i : TileIndex [p.blockInner,p.blockColumns]) :=
  bBase p s i+cur*p.blockInner*p.weightInnerStride
def outputOffset (p : Parameters) (s : State ℝ) (i : TileIndex [p.blockRows,p.blockColumns]) :=
  p.outputRowStride*row p s i.1+p.outputColumnStride*col p s i.2.1
def active (p : Parameters) (s : State ℝ) (i : TileIndex [p.blockRows,p.blockColumns]) : Bool :=
  (row p s i.1<p.rows)&&(col p s i.2.1<p.columns)
def count (p : Parameters) := (p.inner+p.blockInner-1)/p.blockInner
abbrev Accumulator (p : Parameters) := TileIndex [p.blockRows,p.blockColumns] → ℝ

def aValues (p : Parameters) (s : State ℝ) (cur : Nat) :
    TileIndex [p.blockRows,p.blockInner] → ℝ := fun i =>
  if i.2.1.val<p.inner-cur*p.blockInner then (s.mem p.input (aOffset p s cur i)).read .fp32
  else 0

def bValues (p : Parameters) (s : State ℝ) (cur : Nat) :
    TileIndex [p.blockInner,p.blockColumns] → ℝ := fun i =>
  if i.1.val<p.inner-cur*p.blockInner then (s.mem p.weight (bOffset p s cur i)).read .fp32
  else 0

def update (g : Graph) (p : Parameters) (s : State ℝ) (cur : Nat)
    (a : Accumulator p) : Accumulator p :=
  match g with
  | .source | .accumulator => fun i => a i + sumDot (batch:=[]) (aValues p s cur) (bValues p s cur) i

def accumulated (g : Graph) (p : Parameters) (s : State ℝ) : Accumulator p :=
  foldRange (update g p s) 0 (count p) 1 (fun _ => 0)

def outputValues (p : Parameters) (a : Accumulator p) : Accumulator p := a

def written (p : Parameters) (s : State ℝ)
    (a : Accumulator p) : State ℝ :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then t.write p.output (outputOffset p s i)
      (.mk .fp32 (outputValues p a i)) else t) s
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.RealExecution
