import bench.repairs.tritonbench_g.rotary_emb.Contract
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Checked
abbrev Tensor := VeriTile.Bench.Repairs.RotaryEmb.Checked.Tensor

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  D : Nat
  bpos : 0 < B
  hpos : 0 < H
  dpos : 0 < D

def Shape.BH (c : Shape) : Nat := c.B*c.H
def Shape.rows (c : Shape) : Nat := c.BH*c.T*c.D
def Shape.slots (c : Shape) : Nat := c.BH*c.D
def Shape.high (c : Shape) : Nat := c.BH*(c.T+1)*c.D+c.D+127
theorem Shape.bhpos (c : Shape) : 0<c.BH := Nat.mul_pos c.bpos c.hpos

inductive Role where
  | x | g | h0 | o | ht | incoming | dht | dx | dg
  deriving DecidableEq,Repr

def roles : List Role := [.x,.g,.h0,.o,.ht,.incoming,.dht,.dx,.dg]
def name : Role → String
  | .x => "x" | .g => "g" | .h0 => "h0" | .o => "o" | .ht => "ht"
  | .incoming => "do" | .dht => "dht" | .dx => "dx" | .dg => "dg"
def stateRole (r : Role) : Bool := r=.h0 ∨ r=.ht ∨ r=.dht
def shape (c : Shape) (r : Role) : List Nat := if stateRole r then [c.B,c.H,c.D] else [c.B,c.H,c.T,c.D]
def extent (c : Shape) (r : Role) : Nat := if stateRole r then c.slots else c.rows
def present (U S V : Bool) (r : Role) : Bool := match r with
  | .h0 => U | .ht => S | .dht => S && V | _ => Bool.true
def written (r : Role) : Prop := r=.o ∨ r=.ht ∨ r=.dx ∨ r=.dg
def separate (a b : Tensor) : Prop :=
  a.device≠b.device ∨ a.allocationBase+a.allocationBytes ≤ b.allocationBase ∨
    b.allocationBase+b.allocationBytes ≤ a.allocationBase

/-- Successful ordinary allocations and actual observed Tensor metadata.
Forward and backward allocations are fresh at their own creation times. -/
structure Runtime (c : Shape) (U S V : Bool) where
  tensor : Role → Tensor
  device : Nat
  currentDevice : Nat
  freshOutputs : ∀ r q,written r → written q → r≠q → separate (tensor r) (tensor q)
  freshForward : ∀ r,r=.o ∨ r=.ht → ∀ q,q=.x ∨ q=.g ∨ q=.h0 → separate (tensor r) (tensor q)
  freshBackward : ∀ r,r=.dx ∨ r=.dg → ∀ q,q=.x ∨ q=.g ∨ q=.o ∨ q=.ht ∨ q=.h0 ∨ q=.incoming ∨ q=.dht →
    separate (tensor r) (tensor q)

def tensorCheck (c : Shape) (U S V : Bool) (runtime : Runtime c U S V) : Bool :=
  roles.all fun r => if present U S V r then
    let t := runtime.tensor r
    decide (t.shape=shape c r ∧ t.device=runtime.device ∧ t.cuda=Bool.true ∧ t.strided=Bool.true ∧
      t.dtype="float32" ∧ t.elementBytes=4 ∧ t.contiguous=Bool.true ∧
      t.negative=Bool.false ∧ t.conjugate=Bool.false ∧
      (extent c r=0 ∨ (t.base=t.allocationBase+4*t.storageOffset ∧ t.base%4=0 ∧
        4*t.storageOffset+4*extent c r ≤ t.allocationBytes ∧ t.allocationBase+t.allocationBytes ≤ 2^64)))
    else Bool.true

def check (c : Shape) (U S V : Bool) (runtime : Runtime c U S V) : Bool :=
  tensorCheck c U S V runtime && decide (c.BH ≤ 65535 ∧ (c.D+31)/32<2^31 ∧ c.high<2^31 ∧
    runtime.currentDevice=runtime.device) &&
  roles.all (fun r => !present U S V r || decide ((runtime.tensor r).base+4*c.high<2^64))

def tileCheck (BD : Nat) : Bool := decide (BD=32 ∨ BD=64 ∨ BD=128)

end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Checked
