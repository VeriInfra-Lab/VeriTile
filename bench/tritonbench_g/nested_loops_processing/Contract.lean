import VeriTile.Triton.Core.Ast

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Admission

def dimensions (rowIsInt colIsInt : Bool) (rows cols : Int) : Bool :=
  rowIsInt && colIsInt && decide (0 ≤ rows ∧ rows < 2^63 ∧
    0 ≤ cols ∧ cols < 2^63 ∧ rows*cols ≤ 2^62)

def buffer (count base : Nat) : Bool := decide (base%4=0 ∧ base+4*count ≤ 2^64)

end VeriTile.Bench.Repairs.NestedLoopsProcessing.Admission
