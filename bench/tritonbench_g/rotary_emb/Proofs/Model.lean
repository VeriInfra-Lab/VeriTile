import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RotaryEmb

/-- Scalar primitive ports only. Each pair's loads and stores are separate
operations below; no port supplies a tensor, a kernel result, or a frame. -/
structure Ops (α : Type) where
  mul : α → α → α
  sub : α → α → α
  add : α → α → α

/-- The two unmodified source arithmetic expressions. Odd/even coefficient
coordinates are deliberately independent. Default fusion is interpreted by
the provenance-node backend in Native, rather than disabled in Python. -/
def sourcePair {α : Type} (p : Ops α) (x0 x1 c0 s0 c1 s1 : α) : α × α :=
  let a := p.mul x0 c0
  let b := p.mul x1 s0
  let c := p.mul x0 s1
  let d := p.mul x1 c1
  (p.sub a b, p.add c d)

noncomputable def realOps : Ops ℝ := ⟨(· * ·), (· - ·), (· + ·)⟩

inductive Region where
  | q | k | cos | sin | other (identity : Nat)
  deriving DecidableEq, Repr

def dataRegion (key : Bool) : Region := if key then .k else .q

abbrev Memory (α : Type) := Region → Nat → α

/-- Source statement order: six scalar loads, four multiplies, subtraction,
addition, even store, odd store. Both results use the same pre-store loads. -/
def sourceStep {α : Type} (p : Ops α) (region : Region)
    (a0 a1 c0 c1 s0 s1 : Nat) (m : Memory α) : Memory α :=
  let x0 := m region a0
  let x1 := m region a1
  let cv0 := m .cos c0
  let sv0 := m .sin s0
  let cv1 := m .cos c1
  let sv1 := m .sin s1
  let values := sourcePair p x0 x1 cv0 sv0 cv1 sv1
  let afterEven := fun r a => if r = region ∧ a = a0 then values.1 else m r a
  fun r a => if r = region ∧ a = a1 then values.2 else afterEven r a

#axiomsClean sourcePair
end VeriTile.Bench.Repairs.RotaryEmb
