import VeriTile.Triton.Core
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-! Independent mathematical definitions reused from the original task's
Math module. There is no program AST or execution function in this module. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton

inductive UpperBound where
  | finite (value : ℝ)
  | posInf

noncomputable def UpperBound.cap (upper : UpperBound) (x : ℝ) : ℝ :=
  match upper with | .finite value => min x value | .posInf => x

namespace Math
open scoped BigOperators
noncomputable section

structure Layout where
  batches : Nat
  sequence : Nat
  heads : Nat
  chunk : Nat
  BH : Nat
  BC : Nat
  dtBatch : Nat
  dtTime : Nat
  dtHead : Nat
  rateHead : Nat
  biasHead : Nat
  outBatch : Nat
  outChunk : Nat
  outHead : Nat
  outColumn : Nat
  sumBatch : Nat
  sumChunk : Nat
  sumHead : Nat
  sumColumn : Nat

structure Launch where
  batch : Nat
  chunk : Nat
  head : Nat

structure Inputs where
  dt : Nat → ℝ
  rate : Nat → ℝ
  bias : Nat → ℝ

def head (p : Layout) (l : Launch) (i : Nat) := l.head*p.BH+i
def time (p : Layout) (l : Launch) (j : Nat) := l.chunk*p.chunk+j

def dtAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.batch*p.dtBatch+l.chunk*p.chunk*p.dtTime+head p l i*p.dtHead+j*p.dtTime

def outAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.batch*p.outBatch+l.chunk*p.outChunk+head p l i*p.outHead+j*p.outColumn

def sumAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.batch*p.sumBatch+l.chunk*p.sumChunk+head p l i*p.sumHead+j*p.sumColumn

def activeInput (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  head p l i < p.heads ∧ j < min p.chunk (p.sequence-l.chunk*p.chunk)
def activeOutput (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  head p l i < p.heads ∧ j < p.chunk
instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (activeInput p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))
instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (activeOutput p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def softplus (x : ℝ) : ℝ := if x ≤ 20 then Real.log (1+Real.exp x) else x

def prepared (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (i j : Nat) : ℝ :=
  if activeInput p l i j then
    let x := a.dt (dtAddress p l i j) + if hasBias then a.bias (head p l i*p.biasHead) else 0
    upper.cap (max (if useSoftplus then softplus x else x) lower)
  else 0

def rate (p : Layout) (l : Launch) (a : Inputs) (i : Nat) : ℝ :=
  if head p l i < p.heads then a.rate (head p l i*p.rateHead) else 0

def cumulative (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (i : Fin p.BH) (j : Fin p.BC) : ℝ :=
  ∑ t : Fin p.BC, if t.val ≤ j.val then
    prepared p l a hasBias useSoftplus lower upper i.val t.val * rate p l a i.val else 0

end
end Math
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
