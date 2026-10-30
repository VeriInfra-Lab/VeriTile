import bench.tritonbench_g.softmax_triton1.Kernels
import bench.tritonbench_g.softmax_triton1.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Data.List.Range

namespace VeriTile.Bench.Repairs.SoftmaxTriton1

structure Tensor where
  objectId : Nat
  region : VeriTile.Triton.RegionName
  rows : Nat
  cols : Nat
  offset : Nat
  rowStride : Nat
  colStride : Nat
  device : Nat
  negative : Bool
  deriving DecidableEq

def Tensor.index (t : Tensor) (r c : Nat) : Nat := t.offset+r*t.rowStride+c*t.colStride

def Tensor.Injective (t : Tensor) : Prop :=
  ∀ r c r' c', r<t.rows → c<t.cols → r'<t.rows → c'<t.cols →
    t.index r c=t.index r' c' → r=r' ∧ c=c'

structure SameShape (x y : Tensor) : Prop where
  rows : y.rows=x.rows
  cols : y.cols=x.cols
  device : y.device=x.device

structure Access (S C V : Type) where
  cells : S → VeriTile.Triton.RegionName → Nat → C
  decode : C → V

def Access.read {S C V} (a : Access S C V) (s : S) (r : VeriTile.Triton.RegionName) (i : Nat) :=
  a.decode (a.cells s r i)

/-- Every failure retains its actual failed state. Bind supplies no rollback. -/
inductive Result (S A : Type) where
  | ok (value : A) (state : S)
  | error (message : String) (state : S)

def Result.bind {S A B} (r : Result S A) (f : A → S → Result S B) : Result S B :=
  match r with | .ok v s => f v s | .error e s => .error e s

def runRows {S} (step : Nat → S → Result S Unit) : List Nat → S → Result S Unit
  | [],s => .ok () s
  | p::ps,s => (step p s).bind (fun _ t => runRows step ps t)

variable {S C V : Type}

def Inputs (a : Access S C V) (x : Tensor) (values : Nat → Nat → V) (s : S) : Prop :=
  ∀ r c, r<x.rows → c<x.cols → a.read s x.region (x.index r c)=values r c

def OffRows (y : Tensor) (rows : List Nat) (r : VeriTile.Triton.RegionName) (i : Nat) : Prop :=
  ∀ p ∈ rows, ∀ c, c<y.cols → ¬(y.region=r ∧ y.index p c=i)

def FrameRows (a : Access S C V) (y : Tensor) (rows : List Nat) (s t : S) : Prop :=
  ∀ r i, OffRows y rows r i → a.cells t r i=a.cells s r i

/-- This is a *derived* per-program interface, later discharged from the actual
source interpreter and ordinary successful local-call semantics. -/
structure RowLaw (a : Access S C V) (x y : Tensor) (values target : Nat → Nat → V)
    (step : Nat → S → Result S Unit) (invariant : S → Prop) : Prop where
  success : ∀ p s t, p<x.rows → Inputs a x values s → invariant s → step p s=.ok () t →
    (∀ c, c<x.cols → a.read t y.region (y.index p c)=target p c) ∧
    FrameRows a y [p] s t ∧ invariant t

theorem rows_return (a : Access S C V) (x y : Tensor) (values target : Nat → Nat → V)
    (step : Nat → S → Result S Unit) (invariant : S → Prop)
    (law : RowLaw a x y values target step invariant)
    (shape : y.rows=x.rows ∧ y.cols=x.cols) (separate : x.region≠y.region)
    (injective : y.Injective) (rows : List Nat) (valid : ∀ p ∈ rows, p<x.rows)
    (s t : S) (input : Inputs a x values s) (inv : invariant s) (run : runRows step rows s=.ok () t) :
    (∀ p ∈ rows, ∀ c, c<x.cols → a.read t y.region (y.index p c)=target p c) ∧
    FrameRows a y rows s t ∧ invariant t := by
  induction rows generalizing s with
  | nil =>
    simp only [runRows,Result.ok.injEq] at run
    cases run.2
    exact ⟨by simp,(fun _ _ _ => rfl),inv⟩
  | cons p ps ih =>
    cases hs : step p s with
    | error e u => simp [runRows,hs,Result.bind] at run
    | ok v u =>
      cases v
      have first := law.success p s u (valid p (by simp)) input inv hs
      have input' : Inputs a x values u := by
        intro r c hr hc
        have eq := first.2.1 x.region (x.index r c) (by
          intro q hq j hj eq
          exact separate eq.1.symm)
        exact (congrArg a.decode eq).trans (input r c hr hc)
      have tailRun : runRows step ps u=.ok () t := by simpa [runRows,hs,Result.bind] using run
      have rest := ih (fun q hq => valid q (by simp [hq])) u input' first.2.2 tailRun
      refine ⟨?_,?_,rest.2.2⟩
      · intro q hq c hc
        by_cases htail : q∈ps
        · exact rest.1 q htail c hc
        · have eq : q=p := by simpa [htail] using hq
          subst q
          have framed := rest.2.1 y.region (y.index p c) (by
            intro q hq j hj eq
            have same := injective q j p c
              (by rw [shape.1];exact valid q (by simp [hq])) hj
              (by rw [shape.1];exact valid p (by simp)) (by rwa [shape.2]) eq.2
            exact htail (same.1 ▸ hq))
          exact (congrArg a.decode framed).trans (first.1 c hc)
      · intro r i miss
        have mfirst : OffRows y [p] r i := by
          intro q hq c hc
          have eq : q=p := by simpa using hq
          subst q
          exact miss p (by simp) c hc
        have mrest : OffRows y ps r i := fun q hq c hc => miss q (by simp [hq]) c hc
        exact (rest.2.1 r i mrest).trans (first.2.1 r i mfirst)

/-- Caller byte projection is a codec of the *whole backing allocation*. The
index is an arbitrary backing byte, not restricted to the descriptor's view. -/
def backingByte (a : Access S C V) (encode : C → Nat → BitVec 8)
    (s : S) (region : VeriTile.Triton.RegionName) (byte : Nat) : BitVec 8 :=
  encode (a.cells s region (byte/4)) (byte%4)

theorem frame_backing_byte (a : Access S C V) (encode : C → Nat → BitVec 8)
    (y : Tensor) (rows : List Nat) (s t : S) (frame : FrameRows a y rows s t)
    (r : VeriTile.Triton.RegionName) (separate : r≠y.region) (byte : Nat) :
    backingByte a encode t r byte=backingByte a encode s r byte := by
  unfold backingByte
  rw [frame r (byte/4) (by intro p hp c hc eq;exact separate eq.1.symm)]

#axiomsClean rows_return
#axiomsClean frame_backing_byte
end VeriTile.Bench.Repairs.SoftmaxTriton1
