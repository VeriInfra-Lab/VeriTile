import VeriTile.Spec
import VeriTile.Triton.Core.Ast

namespace VeriTile.Triton.FP.DotContraction

/-- FP32 dot, carrier conversion and separate FP32 addition. This is a local
typed syntax template, not a numerical equality or a whole-kernel assumption. -/
def separate (m k n : Nat) (a b accumulator : RegName) : List ComputeStmt :=
  [.assign .real [m,n] accumulator (.compute (.alg .fp32
    (.add .real Broadcast.nil.consSame.consSame (.ref .real [m,n] accumulator)
      (.castFloat .fp32 .real (.dotTyped .fp32 .fp32 .fp32
        (batch := []) (M := m) (K := k) (N := n)
        (.ref .fp32 [m,k] a) (.ref .fp32 [k,n] b))))))]

/-- The same operands with the live accumulator supplied to dot. -/
def seeded (m k n : Nat) (a b accumulator : RegName) : List ComputeStmt :=
  [.assign .real [m,n] accumulator (.compute (.alg .fp32
    (.dotAcc .fp32 .fp32 .real (batch := []) (M := m) (K := k) (N := n)
      (.ref .fp32 [m,k] a)
      (.ref .fp32 [k,n] b) (.ref .real [m,n] accumulator))))]

/-- Only this fixed local source pair can be attached to its experiment.
Evidence is data; use still requires `Spec.AcceptedAtom`, including the external
`EvidenceValidated` premise. Failed and pending evidence provide no permission. -/
def entry (contract : Spec.Contract) (evidence : Spec.Evidence) :
    Spec.RuleEntry ComputeStmt where
  rule := { lhs := separate 32 32 32 "a" "b" "accumulator"
            rhs := seeded 32 32 32 "a" "b" "accumulator"
            contract := contract }
  evidence := evidence

end VeriTile.Triton.FP.DotContraction
