import VeriTile.Triton.Core.Ast

/-! Selected source: the wrapper creates contiguous homogeneous int32 storage.
There is no floating-point payload operation. The three loops, cached tiles,
pointer advances and all 24 stores are retained. The new single-CTA launch,
finite-allocation masks, safe pointer clamping and duplicate-lane suppression
are interpreted separately from the independent ordered-copy mathematics.
-/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
noncomputable section
open scoped Classical

abbrev Lane := Fin 2 × Fin 2
abbrev Tile (α : Type) := Lane → α
abbrev Word := BitVec 32
abbrev Byte := BitVec 8

inductive Register where
  | a1 | a2 | a3
  deriving DecidableEq, Repr

inductive Instruction where
  | advanceA | advanceC
  | load (register : Register)
  | store (register : Register)
  | ctaBarrier
  deriving DecidableEq, Repr

def inner : List Instruction :=
  [.advanceA,.load .a3,.store .a1,.ctaBarrier,.advanceC,.store .a2,.ctaBarrier,.advanceC,.store .a3,.ctaBarrier,.advanceC]
def middle : List Instruction := [.advanceA,.load .a2]++inner++inner
def outer : List Instruction := [.load .a1]++middle++middle++[.advanceA]
def program : List Instruction := outer++outer

structure Registers (α : Type) where
  inputShift : Nat
  outputShift : Nat
  cache : Register → Tile α

def initialRegisters (zero : α) : Registers α := ⟨0,0,fun _ _=>zero⟩

def address (cols shift : Nat) (lane : Lane) : Nat :=
  lane.1.val*cols+(lane.2.val+shift)

def activeLane (cols : Nat) (lane : Lane) : Prop :=
  lane.1.val=0 ∨ 1<cols ∨ lane.2.val=1

def clamp (count offset : Nat) := min offset (count-1)

def inputTile (zero : α) (count cols : Nat) (input : Nat → α) (shift : Nat) : Tile α :=
  fun lane=>if address cols shift lane<count then input (address cols shift lane) else zero

/-- Load/cache/store trace of the literal nested loops after input stability is
proved. This is not the mathematical result and is not an output API. -/
def traceStep (input : Nat → Tile α)
    (state : Registers α × List (Nat × Tile α)) (instruction : Instruction) :
    Registers α × List (Nat × Tile α) :=
  match instruction with
  | .advanceA=>({state.1 with inputShift:=state.1.inputShift+2},state.2)
  | .advanceC=>({state.1 with outputShift:=state.1.outputShift+2},state.2)
  | .load register=>({state.1 with cache:=Function.update state.1.cache register (input state.1.inputShift)},state.2)
  | .store register=>(state.1,state.2++[(state.1.outputShift,state.1.cache register)])
  | .ctaBarrier=>state

def sourceStores (input : Nat → Tile α) (zero : α) : List (Nat × Tile α) :=
  (program.foldl (traceStep input) (initialRegisters zero,[])).2

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
