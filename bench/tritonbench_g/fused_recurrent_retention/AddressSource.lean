import bench.tritonbench_g.fused_recurrent_retention.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
inductive IExpr where
  | atom (value : Int)
  | add (a b : IExpr)
  | sub (a b : IExpr)
  | mul (a b : IExpr)
  deriving Repr

def IExpr.value : IExpr → Int
  | .atom x => x
  | .add a b => a.value+b.value
  | .sub a b => a.value-b.value
  | .mul a b => a.value*b.value

def IExpr.machine : IExpr → BitVec 32
  | .atom x => BitVec.ofInt 32 x
  | .add a b => a.machine+b.machine
  | .sub a b => a.machine-b.machine
  | .mul a b => a.machine*b.machine

def signed32 (x : Int) : Prop := -(2^31) ≤ x ∧ x < 2^31
def signed64 (x : Int) : Prop := -(2^63) ≤ x ∧ x < 2^63

def IExpr.Safe : IExpr → Prop
  | .atom x => signed32 x
  | .add a b => a.Safe ∧ b.Safe ∧ signed32 (a.value+b.value)
  | .sub a b => a.Safe ∧ b.Safe ∧ signed32 (a.value-b.value)
  | .mul a b => a.Safe ∧ b.Safe ∧ signed32 (a.value*b.value)

inductive SourcePhase where | forward | backward deriving DecidableEq
namespace SourceAddress
private def n (v : Nat) : IExpr := .atom v
private def mul := IExpr.mul
private def add := IExpr.add
private def sub := IExpr.sub

/-- Exact signed32 expression roots before pointer arithmetic. In particular,
do/F batch, head and channel terms are *separate* i32 terms added to a pointer;
we do not impose an extra i32 bound on their signed64 pointer sum. -/
def roots (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (iv ik h lk lv : Nat) : List IExpr :=
  let key := add (mul (n ik) (n s.BK)) (n lk)
  let val := add (mul (n iv) (n s.BV)) (n lv)
  let common := [n s.B,n s.H,n s.T,n s.DK,n s.DV,n s.BK,n s.BV,n s.NK,n s.NV,
    mul (n s.B) (n s.H),n iv,n ik,n h,n lk,n lv,n (h%s.H),key,val,
    n (b.q.strides 1),n (b.q.strides 2),n (b.q.strides 3),
    n (b.v.strides 1),n (b.v.strides 2),n (b.v.strides 3),
    mul (n h) (n (b.q.strides 1)),mul (n h) (n (b.v.strides 1))]
  let partialO := mul (add (n h) (mul (mul (n ik) (n s.B)) (n s.H))) (n (b.v.strides 1))
  let partialQ := mul (add (n h) (mul (mul (n iv) (n s.B)) (n s.H))) (n (b.q.strides 1))
  let state := [mul (mul (n h) (n s.DK)) (n s.DV),mul key (n s.DV)]
  match phase with
  | .forward => common++[partialO]++(if seed || extra then state else [])
  | .backward => common++[partialO,partialQ,n (h/s.H),
      n (b.do_.strides 0),n (b.do_.strides 1),n (b.do_.strides 2),n (b.do_.strides 3),
      mul (n (h/s.H)) (n (b.do_.strides 0)),mul (n (h%s.H)) (n (b.do_.strides 1)),
      mul val (n (b.do_.strides 3)),mul (sub (n s.T) (n 1)) (n s.DK),
      mul (sub (n s.T) (n 1)) (n s.DV),mul (sub (n s.T) (n 1)) (n (b.do_.strides 2))]++
      (if seed then state else [])++(if extra then
      [n (b.finalGrad.strides 0),n (b.finalGrad.strides 1),n (b.finalGrad.strides 2),n (b.finalGrad.strides 3),
       mul (n (h/s.H)) (n (b.finalGrad.strides 0)),mul (n (h%s.H)) (n (b.finalGrad.strides 1)),
       mul key (n (b.finalGrad.strides 2)),mul val (n (b.finalGrad.strides 3))] else [])

/-- Exact pointer positions at loop entry/exit r. r=T includes forward dead
increments and reverse dead retreats; T=0 still forms the -1 reverse rebase. -/
def formed (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (iv ik h lk lv r : Nat) : List Int :=
  let key : Int := ik*s.BK+lk
  let val : Int := iv*s.BV+lv
  let q := fun t : Int => (b.q.offset : Int)+h*b.q.strides 1+key+t*s.DK
  let k := fun t : Int => (b.k.offset : Int)+h*b.q.strides 1+key+t*s.DK
  let v := fun t : Int => (b.v.offset : Int)+h*b.v.strides 1+val+t*s.DV
  let do_ := fun t : Int => (b.do_.offset : Int)+(h/s.H)*b.do_.strides 0+
    (h%s.H)*b.do_.strides 1+val*b.do_.strides 3+t*b.do_.strides 2
  let outV := fun t : Int => ((h+ik*s.B*s.H)*b.v.strides 1 : Nat)+val+t*s.DV
  let outK := fun t : Int => ((h+iv*s.B*s.H)*b.q.strides 1 : Nat)+key+t*s.DK
  let state : Int := h*s.DK*s.DV+key*s.DV+val
  let initial := if seed then [b.seed.offset+state] else []
  match phase with
  | .forward => [q r,k r,v r,outV r]++initial++(if extra then [state] else [])
  | .backward =>
      let rev : Int := (s.T : Int)-1-r
      [q 0,k r,v r,do_ r,outK r,q rev,k rev,v rev,do_ rev,outK rev,outV rev]++initial++
        (if extra then [b.finalGrad.offset+(h/s.H)*b.finalGrad.strides 0+
          (h%s.H)*b.finalGrad.strides 1+key*b.finalGrad.strides 2+val*b.finalGrad.strides 3] else [])

end SourceAddress
end VeriTile.Bench.Repairs.FusedRecurrentRetention
