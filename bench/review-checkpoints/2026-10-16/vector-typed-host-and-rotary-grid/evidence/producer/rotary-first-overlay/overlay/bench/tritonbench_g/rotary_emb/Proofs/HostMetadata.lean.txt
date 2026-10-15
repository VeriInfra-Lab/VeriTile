import bench.tritonbench_g.rotary_emb.Correct
import VeriTile.Triton.Launch.Grid
import VeriTile.Triton.Memory.Locality

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
open VeriTile.Triton FullMath

/-- Tensor metadata supplied by trusted shape/stride primitives. These strides
are nonnegative cell strides, as in the retained kernel semantics. -/
structure Tensor3 where
  length : Nat
  heads : Nat
  width : Nat
  batchStride : Nat
  headStride : Nat
  dimStride : Nat
  deriving Repr

structure Tensor2 where
  length : Nat
  width : Nat
  batchStride : Nat
  dimStride : Nat
  deriving Repr

structure Request where
  q : Tensor3
  k : Tensor3
  cos : Tensor2
  sin : Tensor2
  partialRotaryFactor : Float

/-- The trusted Python primitive is `int(n * factor)`, including its floating
multiplication and truncation. `none` is a primitive exception (e.g. NaN/Inf).
Its internals are not postulated as a theorem about kernel execution. -/
abbrev PythonIntProduct := Nat → Float → Option Int

structure Plan where
  request : Request
  signedHeadDim : Int

def Request.assertions (a : Request) : Prop :=
  a.q.length = a.cos.length ∧ a.q.length = a.sin.length ∧
  a.k.length = a.cos.length ∧ a.k.length = a.sin.length

instance (a : Request) : Decidable a.assertions := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

/-- The order is the source's `int` conversion followed by its two asserts.
A negative result is retained here; it is outside the Nat-index kernel domain,
not silently clamped as a claimed successful source execution. -/
def prepare (intProduct : PythonIntProduct) (a : Request) : Option Plan := do
  let d ← intProduct a.q.width a.partialRotaryFactor
  if a.assertions then pure ⟨a,d⟩ else none

def Plan.parameters (c : Plan) : Parameters :=
  ⟨c.request.q.batchStride, c.request.q.headStride, c.request.q.dimStride,
    c.request.k.batchStride, c.request.k.headStride, c.request.k.dimStride,
    c.request.cos.batchStride, c.request.cos.dimStride,
    c.request.sin.batchStride, c.request.sin.dimStride,
    c.request.q.length, c.request.q.heads, c.request.k.heads, 4, 16,
    c.signedHeadDim.toNat⟩

def Plan.headBlocks (c : Plan) : Nat := (c.request.q.heads + 3) / 4
def Plan.seqBlocks (c : Plan) : Nat := (c.request.q.length + 15) / 16

def Plan.grid (c : Plan) : Grid := ⟨[c.headBlocks,c.seqBlocks]⟩
def Plan.numWarps (c : Plan) : Nat := if c.signedHeadDim ≥ 128 then 8 else 4
def Plan.numStages (_ : Plan) : Nat := 1
abbrev Plan.PID (c : Plan) := Fin c.headBlocks × Fin c.seqBlocks
abbrev Plan.Lane (c : Plan) := c.parameters.Lane

def Plan.evenWidth (c : Plan) : Nat := 2 * (c.parameters.dimension / 2)

/-- Actual launch state for this two-axis grid. Local registers are fresh for
this program; memory is threaded from the previous program's final state. -/
def Plan.at (c : Plan) (s : BlockState) (b : c.PID) : BlockState :=
  { s.resetRegs with
    pids := fun ax => if ax=0 then b.1.val else if ax=1 then b.2.val else 0
    numPids := c.grid.toNumPids }

@[simp] theorem at_mem (c : Plan) (s : BlockState) (b : c.PID) :
    (c.at s b).mem = s.mem := rfl
@[simp] theorem at_read (c : Plan) (s : BlockState) (b : c.PID) (r o) :
    (c.at s b).readMem r o = s.readMem r o := rfl

/-- Derived host arguments, asserts, exact trusted conversion and both resource
branches. There is no premise about any kernel output or successful kernel run. -/
theorem prepare_arguments (op : PythonIntProduct) (a : Request) (c : Plan)
    (h : prepare op a = some c) :
    c.request = a ∧ a.assertions ∧
    op a.q.width a.partialRotaryFactor = some c.signedHeadDim ∧
    c.parameters.blockHead = 4 ∧ c.parameters.blockSeq = 16 ∧
    c.grid.dims = [(a.q.heads+3)/4,(a.q.length+15)/16] ∧
    c.numWarps = (if c.signedHeadDim ≥ 128 then 8 else 4) ∧ c.numStages = 1 := by
  cases hd : op a.q.width a.partialRotaryFactor with
  | none => simp [prepare,hd] at h
  | some d =>
    by_cases ha : a.assertions
    · simp [prepare,hd,ha] at h
      subst c
      exact ⟨rfl,ha,rfl,rfl,rfl,rfl,rfl,rfl⟩
    · simp [prepare,hd,ha] at h

/-- Natural dimension agrees with the actual signed host result on the admitted
Nat-index domain. No positive-width or parity restriction is imposed. -/
theorem dimension_binding (c : Plan) (h : 0 ≤ c.signedHeadDim) :
    (c.parameters.dimension : Int) = c.signedHeadDim := Int.toNat_of_nonneg h

end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
