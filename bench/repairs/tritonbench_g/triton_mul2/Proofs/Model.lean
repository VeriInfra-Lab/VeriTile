import bench.repairs.tritonbench_g.triton_mul2.Kernels
import bench.repairs.tritonbench_g.triton_mul2.Contract
import VeriTile.Meta.StatementAudit

/-! Aligned FP32 slots are indexed by byteAddress/4. A program snapshots all
loads before storing; in-place and fresh-output kernels share this semantics. -/
namespace VeriTile.Bench.Repairs.TritonMul2
noncomputable section
abbrev Memory (α : Type) := Nat → α
structure Operations (α : Type) where
  twice : α → α
structure Layout where
  input : Nat
  output : Nat
  count : Nat
  block : Nat
structure Allocated (l : Layout) : Prop where
  input : l.input+l.count ≤ 2^62
  output : l.output+l.count ≤ 2^62
  separation : l.input=l.output ∨ ∀ i j, i<l.count → j<l.count → l.output+i≠l.input+j

def stores (writes : List (Nat × α)) (s : Memory α) : Memory α :=
  writes.foldl (fun t w => Function.update t w.1 w.2) s

def program (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α) : Memory α :=
  stores ((List.range l.block).filterMap fun lane =>
    let offset := pid*l.block+lane
    if offset<l.count then some (l.output+offset, ops.twice (s (l.input+offset))) else none) s

def runPrefix (ops : Operations α) (l : Layout) (initial : Memory α) : Nat → Memory α
  | 0 => initial
  | count+1 => program ops l count (runPrefix ops l initial count)
def launch (ops : Operations α) (l : Layout) (initial : Memory α) : Memory α :=
  runPrefix ops l initial (grid l.count l.block)
def Doubled (twice : α → α) (n : Nat) (input output : Nat → α) : Prop :=
  ∀ i, i<n → output i=twice (input i)
end
end VeriTile.Bench.Repairs.TritonMul2
