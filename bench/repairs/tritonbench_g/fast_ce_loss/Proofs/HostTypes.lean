import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.API
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.FPPlans
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentContract

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API
open CrossEntropy1.Host (Step)
noncomputable section

inductive DType where
  | fp32 | int64
  deriving DecidableEq

structure Descriptor where
  tensor : Tensor
  dtype : DType
  device : Nat
  strides : List Int
  isNeg : Bool
  isConj : Bool
  contiguous : Bool
  pointer : BitVec 64
  allocationBase : BitVec 64
  allocationBytes : Nat
  version : Nat
  deriving DecidableEq

inductive PythonScalar where
  | integer (value : Int)
  | floating (bits : BitVec 64)
  deriving DecidableEq

/-- The bit representation stays in the ABI. The real interpretation is used
only by the mathematical layer, never as a replacement for weak-scalar FP conversion. -/
structure ScalarInterpreter (α : Type) where
  integer : Int → α
  floating : BitVec 64 → α

def ScalarInterpreter.apply {α : Type} (interpret : ScalarInterpreter α) : PythonScalar → α
  | .integer x => interpret.integer x
  | .floating bits => interpret.floating bits

def fpScalar : ScalarInterpreter Floating.Expr := ⟨.pythonInt,.pythonFloat⟩

structure Arguments where
  batch : Nat
  sequence : Nat
  vocab : Nat
  logits : Descriptor
  labels : Descriptor
  options : Options
  capArgument : PythonScalar
  scaleArgument : PythonScalar
  items : Option PythonScalar

def Arguments.shape (args : Arguments) : Shape :=
  let skeleton : Shape := ⟨args.batch,args.sequence,args.vocab,0⟩
  {skeleton with block:=selectedBlock skeleton}

inductive Kernel where
  | forward | chunked | backward
  deriving DecidableEq

inductive Operation where
  | launch (kernel : Kernel) (grid : List Nat) (block warps : Nat) (options : Options)
  | logsumexp
  | addInplace
  | maskIgnored
  | countNonignored
  | sum
  | divide
  | divideBackward
  | expandSumBackward
  | saveContext
  | returnValue

structure Call where
  operation : Operation
  inputs : List Descriptor
  outputs : List Descriptor
  scalars : List PythonScalar
  integers : List Int
  stream : Nat

/-- Ordinary allocation/view/runtime effects. The numerical memory update is
performed by the fixed source interpreter, not returned by these callbacks. -/
structure Runtime (σ : Type) where
  allocate : List Nat → DType → Nat → σ → Step σ Descriptor
  view : Descriptor → List Nat → σ → Step σ Descriptor
  expand : Descriptor → Nat → σ → Step σ Descriptor
  perform : Call → σ → Step σ Unit

structure World (α σ : Type) where
  memory : Memory α
  integer : Address → Int
  native : σ
  calls : List Call
  allocations : List Descriptor

abbrev Action (α σ ρ : Type) := World α σ → Step (World α σ) ρ

def pureAction {α σ ρ : Type} (value : ρ) : Action α σ ρ := fun world => .ok value world

def bindAction {α σ ρ τ : Type} (first : Action α σ ρ) (next : ρ → Action α σ τ) : Action α σ τ :=
  fun world => match first world with
  | .error message failed => .error message failed
  | .ok value after => next value after

instance {α σ : Type} : Monad (Action α σ) where
  pure := pureAction
  bind := bindAction

def read {α σ : Type} : Action α σ (World α σ) := fun world => .ok world world

def allocate {α σ : Type} (runtime : Runtime σ) (shape : List Nat) (dtype : DType) (device : Nat) :
    Action α σ Descriptor := fun world =>
  match runtime.allocate shape dtype device world.native with
  | .error message native => .error message {world with native:=native}
  | .ok tensor native => .ok tensor {world with native:=native,allocations:=world.allocations++[tensor]}

def view {α σ : Type} (runtime : Runtime σ) (tensor : Descriptor) (shape : List Nat) : Action α σ Descriptor :=
  fun world => match runtime.view tensor shape world.native with
  | .error message native => .error message {world with native:=native}
  | .ok output native => .ok output {world with native:=native}

def expand {α σ : Type} (runtime : Runtime σ) (tensor : Descriptor) (rows : Nat) : Action α σ Descriptor :=
  fun world => match runtime.expand tensor rows world.native with
  | .error message native => .error message {world with native:=native}
  | .ok output native => .ok output {world with native:=native}

def perform {α σ : Type} (runtime : Runtime σ) (call : Call)
    (floats : Memory α → Memory α := id) (integers : (Address → Int) → Address → Int := id) : Action α σ Unit :=
  fun world => match runtime.perform call world.native with
  | .error message native => .error message {world with native:=native}
  | .ok _ native => .ok () {world with
      memory:=floats world.memory
      integer:=integers world.integer
      native:=native
      calls:=world.calls++[call]}

def writeInteger (tensor : Descriptor) (value : Int) (memory : Address → Int) : Address → Int :=
  fun cell => if cell=addr tensor.tensor 0 then value else memory cell

structure SavedContext where
  logits : Descriptor
  lse : Descriptor
  labels : Descriptor
  options : Options

structure ForwardReturn where
  scalar : Descriptor
  customLoss : Descriptor
  context : SavedContext
  chunks : Descriptor
  sum : Descriptor
  count : Option Descriptor
  args : Arguments

structure WholeReturn where
  forward : ForwardReturn
  customBackward : List (Option Tensor)
  publicGradient : Descriptor
  upstream : Descriptor

/-- Complete entry, intermediate allocation and dispatch descriptors form the
ordinary resident key. Scalar values are actual raw host representations. -/
structure CompleteABI (σ : Type) where
  args : Arguments
  initial : σ
  scalarUpstream : Descriptor
  returned : WholeReturn
  calls : List Call
  compiler : String
  target : String
  stream : Nat
  enableFPFusion : Bool
  allocationTrace : List Descriptor

end
end VeriTile.Bench.Repairs.FastCeLoss.Host
