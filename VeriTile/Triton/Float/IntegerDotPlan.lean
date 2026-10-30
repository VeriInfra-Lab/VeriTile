import VeriTile.Triton.Float.Structural
import Mathlib.Data.List.Perm.Basic

/-! Finite source-selected INT8 reduction plans. Every instruction port has
fixed arity 32; scalar tails and explicit zero holes remain visible data.
No plan contains memory access, an executor, or a callback for a whole dot. -/
namespace VeriTile.Triton.FP.Structural.IntegerDot
noncomputable section

inductive WgmmaN where
  | n8 | n16 | n32 | n64 | n128 | n256
  deriving DecidableEq, Repr

inductive ISA where
  | mmaM16N8K32Satfinite
  | wgmmaM64K32 (n : WgmmaN)
  deriving DecidableEq, Repr

structure Instructions (I : Type) where
  zero : I
  product8 : BitVec 8 → BitVec 8 → I
  add : I → I → I
  mma32 : ISA → (Fin 32 → BitVec 8) → (Fin 32 → BitVec 8) → I → I

inductive Micro (k : Nat) where
  | scalar (lane : Fin k)
  | instruction (kind : ISA) (lanes : Fin 32 → Option (Fin k))

def Micro.terms : Micro k → List (Fin k)
  | .scalar lane => [lane]
  | .instruction _ lanes => (List.ofFn lanes).filterMap id

def Micro.eval (a : Instructions I) (left right : Fin k → BitVec 8) (carry : I) : Micro k → I
  | .scalar lane => a.add carry (a.product8 (left lane) (right lane))
  | .instruction kind lanes => a.mma32 kind
      (fun i => ((lanes i).map left).getD 0)
      (fun i => ((lanes i).map right).getD 0) carry

/-- Exact source operand coverage, with the selected fusion boundary retained.
The primitive model does not imply order invariance or a zero-padding law. -/
structure Plan (k : Nat) where
  operations : List (Micro k)
  complete : (operations.flatMap Micro.terms).Perm (List.finRange k)
  fused : Bool

def Plan.eval (plan : Plan k) (a : Instructions I)
    (left right : Fin k → BitVec 8) (old : I) : I :=
  if plan.fused then plan.operations.foldl (fun carry op => op.eval a left right carry) old
  else a.add old (plan.operations.foldl (fun carry op => op.eval a left right carry) a.zero)

/-- Scalar representation conversion is explicit. In particular signed-word
encoding need not be a left inverse on arbitrary mathematical Int values. -/
structure Arithmetic where
  normalize : Int → Int
  product8 : BitVec 8 → BitVec 8 → Int
  add : Int → Int → Int
  mma32 : ISA → (Fin 32 → BitVec 8) → (Fin 32 → BitVec 8) → Int → Int

def Arithmetic.instructions (a : Arithmetic) : Instructions Int where
  zero := 0
  product8 := a.product8
  add := a.add
  mma32 := a.mma32

/-- The fused case consumes the actual encoded source carry. The non-fused
case retains its original final scalar add after the zero-seeded dot. -/
def Plan.evalInt (plan : Plan k) (a : Arithmetic)
    (left right : Fin k → BitVec 8) (old : Int) : Int :=
  if plan.fused then
    plan.operations.foldl (fun carry op => op.eval a.instructions left right carry) (a.normalize old)
  else a.add old (plan.operations.foldl (fun carry op => op.eval a.instructions left right carry) 0)

end
end VeriTile.Triton.FP.Structural.IntegerDot
