import bench.tritonbench_g.rmsnorm_implementation.Kernels

/-! ## Whole-wrapper observed FP32 source graph

The source-level typed graph above is unchanged. The additional two-pass machine
below records the compiler-observed FP32 graph explicitly: one fused square-add
per active or padded lane, a complete scalar-add reduction tree, FP32 division,
epsilon addition, square root, and second-pass division/multiplication. It makes
no equality assumption between the original split arithmetic and native FMA.
Compiler/source correspondence is an explicit external boundary of this route.
-/
namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section

abbrev Memory (α : Type) := Nat → Nat → α

structure Tensor where
  shape : List Nat
  stride : Nat → Nat
  allocation : Nat
  base : Nat
  device : Nat
  dtype : Storage

structure Parameters where
  batches : Nat
  rows : Nat
  width : Nat
  input : Tensor
  weights : Tensor
  output : Tensor
  eps : ℝ

def inputOffset (p : Parameters) (b m k : Nat) : Nat :=
  p.input.base + b*p.input.stride 0 + m*p.input.stride 1 + k*p.input.stride 2

def weightOffset (p : Parameters) (k : Nat) : Nat :=
  p.weights.base + k*p.weights.stride 0

def outputOffset (p : Parameters) (b m k : Nat) : Nat :=
  p.output.base + b*p.output.stride 0 + m*p.output.stride 1 + k*p.output.stride 2

def trips (p : Parameters) : Nat := (p.width+4095)/4096

def columns (p : Parameters) : List Nat :=
  (List.range (trips p)).flatMap (fun t => (List.range 4096).map (fun j => t*4096+j))

def programGrid (p : Parameters) : List (Nat × Nat) :=
  (List.range p.batches).flatMap (fun b => (List.range p.rows).map (fun m => (b,m)))

inductive SumTree where
  | leaf (lane : Fin 4096)
  | add (left right : SumTree)

def SumTree.leaves : SumTree → List (Fin 4096)
  | .leaf j => [j]
  | .add a b => a.leaves ++ b.leaves

def SumTree.eval (add : α → α → α) (xs : Fin 4096 → α) : SumTree → α
  | .leaf j => xs j
  | .add a b => add (a.eval add xs) (b.eval add xs)

structure Reduction where
  tree : SumTree
  complete : tree.leaves.Perm (List.finRange 4096)

/-- Each field is one scalar primitive. No operation accepts a row or memory,
chooses a source index, or supplies any complete-row result. -/
structure Numerics (α : Type) where
  literal : ℝ → α
  squareAcc : α → α → α
  add : α → α → α
  divideByWidth : α → Nat → α
  addEpsilon : α → ℝ → α
  div : α → α → α
  sqrt : α → α
  mul : α → α → α

/-- Source first-loop body: physical input address, true zero input padding,
explicit FP32 storage/cast identity on the selected native graph, one FMA. -/
def varianceStep (Q : Numerics α) (p : Parameters) (b m t : Nat)
    (memory : Memory α) (acc : Fin 4096 → α) : Fin 4096 → α := fun j =>
  let k := t*4096+j.val
  let x := if k < p.width then memory p.input.allocation (inputOffset p b m k)
           else Q.literal 0
  Q.squareAcc x (acc j)

/-- Direct physical write; output masks govern writes, never input values. -/
def store (memory : Memory α) (allocation offset : Nat) (value : α) : Memory α :=
  fun a i => if a = allocation ∧ i = offset then value else memory a i

/-- The omitted weight padding is retained as an arbitrary unobservable word.
The first input load uses zero padding. Both arithmetic operations occur in the
source body before the masked write. -/
def outputLane (Q : Numerics α) (p : Parameters) (b m k : Nat)
    (std : α) (undefinedWeight : Nat → α) (memory : Memory α) : Memory α :=
  let active := k < p.width
  let weight := if active then memory p.weights.allocation (weightOffset p k)
                else undefinedWeight k
  let x := if active then memory p.input.allocation (inputOffset p b m k)
           else Q.literal 0
  let out := Q.mul (Q.div x std) weight
  if active then store memory p.output.allocation (outputOffset p b m k) out else memory

def outputPass (Q : Numerics α) (p : Parameters) (b m : Nat)
    (std : α) (undefinedWeight : Nat → α) : List Nat → Memory α → Memory α
  | [], memory => memory
  | k::ks, memory => outputPass Q p b m std undefinedWeight ks
      (outputLane Q p b m k std undefinedWeight memory)

/-- Complete compiler-observed source program. Both original 4096-wide loops
remain: first updates all accumulator lanes, second traverses all padded lanes
and performs the original loads/division/multiplication/masked store. -/
def executeRow (Q : Numerics α) (reduction : Reduction) (p : Parameters)
    (b m : Nat) (undefinedWeight : Nat → α) (memory : Memory α) : Memory α :=
  let variance := (List.range (trips p)).foldl
    (fun acc t => varianceStep Q p b m t memory acc) (fun _ => Q.literal 0)
  let sum := reduction.tree.eval Q.add variance
  let mean := Q.divideByWidth sum p.width
  let std := Q.sqrt (Q.addEpsilon mean p.eps)
  outputPass Q p b m std undefinedWeight (columns p) memory

def executeGrid (Q : Numerics α) (reduction : Reduction) (p : Parameters)
    (undefinedWeight : Nat → α) : List (Nat × Nat) → Memory α → Memory α
  | [], memory => memory
  | (b,m)::rest, memory => executeGrid Q reduction p undefinedWeight rest
      (executeRow Q reduction p b m undefinedWeight memory)

/-- Independent logical recurrence; it has no memory, address or interpreter
argument. Inactive lanes contribute the source's zero through the same FMA. -/
def laneAccumulator (Q : Numerics α) (width : Nat) (x : Nat → α) :
    Nat → Fin 4096 → α
  | 0, _ => Q.literal 0
  | t+1,j => Q.squareAcc (if t*4096+j.val < width then x (t*4096+j.val) else Q.literal 0)
      (laneAccumulator Q width x t j)

def logicalStd (Q : Numerics α) (reduction : Reduction) (width : Nat)
    (eps : ℝ) (x : Nat → α) : α :=
  Q.sqrt (Q.addEpsilon (Q.divideByWidth (reduction.tree.eval Q.add
    (laneAccumulator Q width x ((width+4095)/4096))) width) eps)

def logicalValue (Q : Numerics α) (reduction : Reduction) (width : Nat)
    (eps : ℝ) (x w : Nat → α) (k : Nat) : α :=
  Q.mul (Q.div (x k) (logicalStd Q reduction width eps x)) (w k)

/-- Independent mathematical RMSNorm formula, over original tensor coordinates. -/
def mathematical (width : Nat) (eps : ℝ) (x w : Nat → ℝ) (k : Nat) : ℝ :=
  (x k / Real.sqrt ((∑ j : Fin width, (x j.val)^2)/(width : ℝ)+eps)) * w k

end
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
