import Mathlib.Data.Int.Basic
import Mathlib.Data.BitVec
import Mathlib.Data.Finset.Basic

/-! Retained finite-width event interpretation of the separate repair. Metadata cells are signed int32; payload cells are raw
FP32 words. Addresses are byte addresses. This is the concrete copy program,
with fixed block 256, signed-i64 loop/lane arithmetic and u64 pointers. -/
namespace VeriTile.Bench.Repairs.VarLenCopy
noncomputable section

def rounds (length : Int) (block : Nat) : Nat := (length.toNat + block - 1) / block

abbrev Word := BitVec 32
abbrev Memory := Nat → Word
abbrev Event := Nat × Nat -- destination byte address, source byte address

def addr (base offset : Nat) := base + 4 * offset
def word64 (n : Nat) := n % 2^64
def signed64 (n : Int) := (BitVec.ofInt 64 n).toInt
def pointer (base : Nat) (offset : Int) :=
  (BitVec.ofInt 64 ((base : Int) + 4 * offset)).toNat

structure Layout where
  oldStarts : Nat
  lengths : Nat
  newStarts : Nat
  source : Nat
  output : Nat
  count : Nat
  sourceSize : Nat
  outputSize : Nat

structure Segment where
  oldStart : Word
  length : Word
  newStart : Word
  deriving DecidableEq

def Segment.old (a : Segment) := a.oldStart.toInt.toNat
def Segment.len (a : Segment) := a.length.toInt.toNat
def Segment.dest (a : Segment) := a.newStart.toInt.toNat

def loadSegment (l : Layout) (s : Memory) (pid : Nat) : Segment :=
  ⟨s (pointer l.oldStarts (signed64 pid)), s (pointer l.lengths (signed64 pid)),
   s (pointer l.newStarts (signed64 pid))⟩

/-- One concrete native lane. The conditional suppresses both load and store. -/
def laneEvent (l : Layout) (a : Segment) (round lane : Nat) : Option Event :=
  let i := signed64 ((round * 256 : Nat) : Int)
  let offset := signed64 (i + signed64 lane)
  if offset < signed64 a.length.toInt then
    some (pointer l.output (signed64 (signed64 a.newStart.toInt + offset)),
          pointer l.source (signed64 (signed64 a.oldStart.toInt + offset)))
  else none

def events (l : Layout) (a : Segment) : List Event :=
  (List.range (rounds a.length.toInt 256)).flatMap fun round =>
    (List.range 256).filterMap (laneEvent l a round)

def copyEvent (s : Memory) (e : Event) : Memory :=
  Function.update s e.1 (s e.2)

def execute (xs : List Event) (s : Memory) : Memory := xs.foldl copyEvent s

def program (l : Layout) (pid : Nat) (s : Memory) : Memory :=
  execute (events l (loadSegment l s pid)) s

def launch (l : Layout) (s : Memory) : Memory :=
  (List.range l.count).foldl (fun t pid => program l pid t) s

end
end VeriTile.Bench.Repairs.VarLenCopy
