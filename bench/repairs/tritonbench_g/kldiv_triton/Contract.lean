import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Binding

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Contiguous
open Host (blockSize modeCode block_positive)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
open scoped BigOperators
noncomputable section
set_option autoImplicit false

/-- Independent pointwise formula, with the selected raw-target zero extension. -/
def loss (flag : Bool) (prediction target : ℝ) : ℝ :=
  if flag then Real.exp target * (target-prediction)
  else if target=0 then 0 else target*(Real.log target-prediction)

/-- Reduced modes write one unnormalized sum per chunk. Python's final sum
and mean/batchmean normalization are outside this kernel target. -/
def forwardTarget (flag : Bool) (mode : Math.Mode) (cols : Nat)
    (prediction target : Nat → ℝ) (column : Nat) : ℝ :=
  if mode=.none then loss flag (prediction column) (target column)
  else ∑ lane : Fin (blockSize cols),
    if column*blockSize cols+lane.val<cols then
      loss flag (prediction (column*blockSize cols+lane.val))
        (target (column*blockSize cols+lane.val)) else 0

/-- The backward kernel writes only the unscaled derivative. -/
def backwardTarget (flag : Bool) (target : ℝ) : ℝ :=
  if flag then -Real.exp target else -target

def outputCols (mode : Math.Mode) (cols : Nat) : Nat :=
  if mode=.none then cols else rounds cols (blockSize cols)

/-- Exact selected forward admission and ordinary bindings of its existing
fresh zero allocation. No new input restriction or numeric premise is added. -/
structure ForwardDomain (a b o : Tensor) (prediction target output : Matrix)
    (mode : Math.Mode) : Prop where
  admitted : ForwardContract a b
  predictionBinding : MatrixBinding a prediction
  targetBinding : MatrixBinding b target
  outputBinding : MatrixBinding o output
  outputAdmitted : TensorContract o
  outputDevice : o.metadata.device=a.metadata.device
  outputRows : output.rows=prediction.rows
  outputCols : output.cols=Kernel.outputCols mode prediction.cols
  outputUnit : output.colStride=1
  freshPrediction : prediction.view.storage.separate output.view.storage
  freshTarget : target.view.storage.separate output.view.storage

/-- Exact selected backward checks, including the original whole-storage
separation check for nonempty destinations. Cotangent processing is external. -/
structure BackwardDomain (a b g : Tensor) (destination target : Matrix)
    (gradient : View) : Prop where
  admitted : BackwardContract a b g
  destinationBinding : MatrixBinding a destination
  targetBinding : MatrixBinding b target
  gradientBinding : ViewBinding g gradient

/-- Successful dispatch uses representable native program IDs. This is the
ordinary launcher boundary, not an additional Python admission guard. -/
abbrev Dispatch := L2NormBwd.Integer.GridCompatible

def forwardLayout (prediction target output : Matrix) (flag : Bool) (mode : Math.Mode) :
    Physical.ForwardLayout :=
  ⟨prediction,target,output,blockSize prediction.cols,flag,modeCode mode⟩

def backwardLayout (destination target : Matrix) (flag : Bool) : Physical.BackwardLayout :=
  ⟨target,destination,blockSize destination.cols,flag⟩

/-- Logical contents are read from the actual initial physical View keys. -/
def row {M α β : Type} (ops : Ops M α β) (neg : Negation α) (memory : M)
    (m : Matrix) (r c : Nat) : α := logical ops neg memory m.view (r*m.cols+c)

abbrev python_forward_checks_iff_preconditions := Contiguous.python_forward_checks_iff
abbrev python_backward_checks_iff_preconditions := Contiguous.python_backward_checks_iff

#specNonCircular loss avoiding [forwardSource,backwardSource]
#specNonCircular forwardTarget avoiding [forwardSource,Tile.loss,Tile.chunkSum,Storage.forwardValue]
#specNonCircular backwardTarget avoiding [backwardSource,Tile.gradient,Storage.backwardValue]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel
