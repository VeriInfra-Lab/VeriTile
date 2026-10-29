import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelSource
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Arithmetic
import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelValues
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

/-- Structural dot interpretation. Concrete public interpretations below
construct this from a Real sum or fixed instruction lanes; they do not assume
an equality between an arbitrary whole-dot result and a specification. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → (left right output : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) →
    (TileIndex (batch++[m,n]) → α)

def engine (M : Algebra α) (D : Dot α) : Algebra α :=
  let base : Algebra α := {M with
    dotTyped := fun prec l r out options {batch} {m k n} a b =>
      some (fun i => ofFloat out (D.apply prec l r out options
        (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i))}
  base.withDefaultPrecision .fp32

def gridRows (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def gridCols (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def width (p : Parameters) := p.groupRows*gridCols p
def groupIdx (p : Parameters) (s : State α) := s.pids 0/width p
def groupSize (p : Parameters) (s : State α) :=
  if gridRows p-groupIdx p s*p.groupRows<p.groupRows then
    gridRows p-groupIdx p s*p.groupRows else p.groupRows
def blockRow (p : Parameters) (s : State α) := groupIdx p s*p.groupRows+s.pids 0%groupSize p s
def blockCol (p : Parameters) (s : State α) := s.pids 0%width p/groupSize p s
def row (p : Parameters) (s : State α) (i : Fin p.blockRows) := blockRow p s*p.blockRows+i.val
def col (p : Parameters) (s : State α) (j : Fin p.blockColumns) := blockCol p s*p.blockColumns+j.val
def aOffset (p : Parameters) (s : State α) (cur : Nat) (i : TileIndex [p.blockRows,p.blockInner]) :=
  (row p s i.1%p.rows)*p.inputRowStride+i.2.1.val*p.inputInnerStride+cur*p.blockInner*p.inputInnerStride
def bOffset (p : Parameters) (s : State α) (cur : Nat) (i : TileIndex [p.blockInner,p.blockColumns]) :=
  i.1.val*p.weightInnerStride+(col p s i.2.1%p.columns)*p.weightColStride+cur*p.blockInner*p.weightInnerStride
def outputOffset (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,p.blockColumns]) :=
  row p s i.1*p.outputRowStride+col p s i.2.1*p.outputColStride
def savedOffset (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,p.blockColumns]) :=
  row p s i.1*p.savedRowStride+col p s i.2.1*p.savedColStride
def active (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,p.blockColumns]) : Bool :=
  (row p s i.1<p.rows)&&(col p s i.2.1<p.columns)
def read (mode : Mode) (s : State α) (r : RegionName) (o : Nat) : α :=
  toFloat (dtype mode) ((s.mem r o).read (dtype mode).toTileDType)
abbrev Accumulator (α : Type) (p : Parameters) := TileIndex [p.blockRows,p.blockColumns] → α

def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def paddedZero (M : Algebra α) (mode : Mode) :=
  M.cast (some .fp32) .real (dtype mode) (M.literal (some .fp32) .real 0.0)
def biasValue (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) (j : Fin p.blockColumns) :=
  M.cast (some .fp32) (dtype mode) .real
    (if col p s j%p.columns<p.columns then read mode s p.bias (col p s j%p.columns) else paddedZero M mode)
def initial (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) : Accumulator α p :=
  fun i => if p.hasBias then M.binary (some .fp32) .real .add (zero M) (biasValue M mode p s i.2.1) else zero M
def aValues (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) (cur : Nat) :
    TileIndex [p.blockRows,p.blockInner] → α := fun i =>
  if p.inner%p.blockInner=0 ∨ (i.2.1.val:Int)<(p.inner:Int)-(cur*p.blockInner:Int) then
    read mode s p.input (aOffset p s cur i) else paddedZero M mode
def bValues (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) (cur : Nat) :
    TileIndex [p.blockInner,p.blockColumns] → α := fun i =>
  if p.inner%p.blockInner=0 ∨ (i.1.val:Int)<(p.inner:Int)-(cur*p.blockInner:Int) then
    read mode s p.weight (bOffset p s cur i) else paddedZero M mode
def update (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters) (s : State α)
    (cur : Nat) (acc : Accumulator α p) : Accumulator α p :=
  fun i => M.binary (some .fp32) .real .add (acc i)
    (M.cast (some .fp32) .fp32 .real
      (D.apply (some .fp32) (dtype mode) (dtype mode) .fp32 {} (batch:=[])
        (aValues M mode p s cur) (bValues M mode p s cur) i))
def accumulated (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters) (s : State α) : Accumulator α p :=
  foldN (update M D mode p s) (ceilDiv p.inner p.blockInner) (initial M mode p s)

/-- Source-order helper expressions, before any compiler contraction. -/
def activation (M : Algebra α) (name : Option String) (x : α) : α :=
  let half := M.literal (some .fp32) .real 0.5
  let one := M.literal (some .fp32) .real 1.0
  let add := M.binary (some .fp32) .real .add
  let mul := M.binary (some .fp32) .real .mul
  if name=some "tanh" then M.unary (some .fp32) .tanh x
  else if name=some "gelu" then mul (mul x half) (add one
    (M.unary (some .fp32) .erf (M.binary (some .fp32) .real .div x (M.literal (some .fp32) .real (Real.sqrt 2)))))
  else if name=some "fast_gelu" then mul (mul half x) (add (M.literal (some .fp32) .real 1)
    (M.unary (some .fp32) .tanh (mul (M.literal (some .fp32) .real (Real.sqrt (2.0/Real.pi)))
      (add x (mul (mul (mul (M.literal (some .fp32) .real 0.044715) x) x) x)))))
  else if name=some "relu" then M.binary (some .fp32) .real .max (zero M) x
  else x

def outputCell (M : Algebra α) (mode : Mode) (x : α) : Cell α :=
  .mk (dtype mode).toTileDType (ofFloat (dtype mode) (M.cast (some .fp32) .real (dtype mode) x))
def scatter (M : Algebra α) (mode : Mode) (p : Parameters) (geometry base : State α)
    (region : RegionName) (off : TileIndex [p.blockRows,p.blockColumns] → Nat) (values : Accumulator α p) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p geometry i then t.write region (off i) (outputCell M mode (values i)) else t) base
def written (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) (acc : Accumulator α p) : State α :=
  let saved := if p.saveInputs then scatter M mode p s s p.saved (savedOffset p s) acc else s
  scatter M mode p s saved p.output (outputOffset p s) (fun i => activation M p.activation (acc i))

end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelValues
