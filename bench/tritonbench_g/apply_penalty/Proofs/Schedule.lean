import bench.tritonbench_g.apply_penalty.Kernels
import bench.tritonbench_g.apply_penalty.Proofs.Mathematical_Kernels
import bench.tritonbench_g.apply_penalty.Contract

/-! Source definitions only. Widened nonnegative signed indices erase to the
existing source. `rowLanes` retains loads from the incoming mutable logits,
both arithmetic arms, integer-count conversion, masks and in-place stores. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty
open VeriTile.Triton

abbrev realKernel := VeriTile.Bench.TritonBenchG.ApplyPenalty.apply_penalty

structure ScalarOps (Word : Type) where
  zero : Word
  fromInt32 : BitVec 32 → Word
  positive : Word → Bool
  divide : Word → Word → Word
  multiply : Word → Word → Word
  subtract : Word → Word → Word
  /-- Local lowering of `selected - converted_count * frequency`. A fused
  backend and a separately rounded backend must supply different operations. -/
  subtractProduct : Word → Word → Word → Word

structure Inputs (Word : Type) where
  logits : Nat → Word
  presence : Nat → Word
  frequency : Nat → Word
  repetition : Nat → Word

structure Registers (Word : Type) where
  quotient : Word
  product : Word
  selected : Word
  converted : Word
  frequency : Word
  result : Word

def arithmetic {Word : Type} (ops : ScalarOps Word) (x p f r : Word)
    (count : BitVec 32) : Registers Word :=
  let quotient := ops.divide x r
  let product := ops.multiply x r
  let selected := if ops.positive x then quotient else product
  let converted := ops.fromInt32 count
  let frequency := ops.subtractProduct selected converted f
  ⟨quotient, product, selected, converted, frequency, ops.subtract frequency p⟩

def rowLanes {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (memory : Nat → Word) (b lane : Nat) : Registers Word :=
  let offset := a.start b + lane
  let active := offset < a.stop b
  let token := if active then a.token offset else 0
  let count := if active then a.counts offset else 0
  let x := if active then memory (b*a.vocab+token) else ops.zero
  arithmetic ops x (input.presence b) (input.frequency b) (input.repetition b) count

def rowEvents (a : Arguments) (b : Nat) : List Nat :=
  (List.range a.block).filter fun lane => a.start b + lane < a.stop b

def store {Word : Type} (memory : Nat → Word) (address : Nat) (value : Word) : Nat → Word :=
  fun q => if q = address then value else memory q

def rowRun {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (memory : Nat → Word) (b : Nat) : Nat → Word :=
  (rowEvents a b).foldl (fun current lane =>
    store current (b*a.vocab+a.token (a.start b+lane))
      (rowLanes ops a input memory b lane).result) memory

def runRows {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) : List Nat → (Nat → Word) → (Nat → Word)
  | [], memory => memory
  | b :: rest, memory => runRows ops a input rest (rowRun ops a input memory b)

def gridRun {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (order : List Nat) : Nat → Word :=
  runRows ops a input order input.logits

noncomputable def wrapperRun {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) : Except InputError (Nat → Word) := do
  checkInputs a
  return gridRun ops a input (List.range a.batches)

end VeriTile.Bench.Repairs.ApplyPenalty
