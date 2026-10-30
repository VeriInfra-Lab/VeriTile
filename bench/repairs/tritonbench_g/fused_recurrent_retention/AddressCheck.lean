import bench.repairs.tritonbench_g.fused_recurrent_retention.AddressSource
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission

/-- Structural integer expression with eight bounded index variables. The
magnitude checker is executable and examines every source subexpression. -/
inductive SExpr where
  | const (value : Int)
  | var (index : Nat)
  | add (a b : SExpr)
  | sub (a b : SExpr)
  | mul (a b : SExpr)

def SExpr.instantiate (env : Nat → Int) : SExpr → IExpr
  | .const x => .atom x
  | .var i => .atom (env i)
  | .add a b => .add (a.instantiate env) (b.instantiate env)
  | .sub a b => .sub (a.instantiate env) (b.instantiate env)
  | .mul a b => .mul (a.instantiate env) (b.instantiate env)

def SExpr.radius (limits : Nat → Nat) : SExpr → Nat
  | .const x => x.natAbs
  | .var i => limits i
  | .add a b | .sub a b => a.radius limits+b.radius limits
  | .mul a b => a.radius limits*b.radius limits

def SExpr.check (limits : Nat → Nat) (e : SExpr) : Bool :=
  decide (e.radius limits<2^31) && match e with
  | .const _ | .var _ => Bool.true
  | .add a b | .sub a b | .mul a b => a.check limits && b.check limits

def limits (s : Shape) : Nat → Nat
  | 0 => s.NV-1 | 1 => s.NK-1 | 2 => s.BH-1 | 3 => s.BK-1
  | 4 => s.BV-1 | 5 => s.B-1 | 6 => s.H-1 | 7 => s.T | _ => 0

def env (s : Shape) (iv ik h lk lv r : Nat) : Nat → Int
  | 0 => iv | 1 => ik | 2 => h | 3 => lk | 4 => lv
  | 5 => ((h/s.H : Nat) : Int) | 6 => ((h%s.H : Nat) : Int) | 7 => r | _ => 0

def sNat (value : Nat) : SExpr := .const value
private def mul := SExpr.mul
private def add := SExpr.add
private def sub := SExpr.sub

def roots (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    : List SExpr :=
  let key := add (mul (SExpr.var 1) (sNat s.BK)) (SExpr.var 3)
  let val := add (mul (SExpr.var 0) (sNat s.BV)) (SExpr.var 4)
  let common := [sNat s.B,sNat s.H,sNat s.T,sNat s.DK,sNat s.DV,sNat s.BK,sNat s.BV,sNat s.NK,sNat s.NV,
    mul (sNat s.B) (sNat s.H),SExpr.var 0,SExpr.var 1,SExpr.var 2,SExpr.var 3,SExpr.var 4,SExpr.var 6,key,val,
    sNat (b.q.strides 1),sNat (b.q.strides 2),sNat (b.q.strides 3),
    sNat (b.v.strides 1),sNat (b.v.strides 2),sNat (b.v.strides 3),
    mul (SExpr.var 2) (sNat (b.q.strides 1)),mul (SExpr.var 2) (sNat (b.v.strides 1))]
  let partialO := mul (add (SExpr.var 2) (mul (mul (SExpr.var 1) (sNat s.B)) (sNat s.H))) (sNat (b.v.strides 1))
  let partialQ := mul (add (SExpr.var 2) (mul (mul (SExpr.var 0) (sNat s.B)) (sNat s.H))) (sNat (b.q.strides 1))
  let state := [mul (mul (SExpr.var 2) (sNat s.DK)) (sNat s.DV),mul key (sNat s.DV)]
  match phase with
  | .forward => common++[partialO]++(if seed || extra then state else [])
  | .backward => common++[partialO,partialQ,SExpr.var 5,
      sNat (b.do_.strides 0),sNat (b.do_.strides 1),sNat (b.do_.strides 2),sNat (b.do_.strides 3),
      mul (SExpr.var 5) (sNat (b.do_.strides 0)),mul (SExpr.var 6) (sNat (b.do_.strides 1)),
      mul val (sNat (b.do_.strides 3)),mul (sub (sNat s.T) (sNat 1)) (sNat s.DK),
      mul (sub (sNat s.T) (sNat 1)) (sNat s.DV),mul (sub (sNat s.T) (sNat 1)) (sNat (b.do_.strides 2))]++
      (if seed then state else [])++(if extra then
      [sNat (b.finalGrad.strides 0),sNat (b.finalGrad.strides 1),sNat (b.finalGrad.strides 2),sNat (b.finalGrad.strides 3),
       mul (SExpr.var 5) (sNat (b.finalGrad.strides 0)),mul (SExpr.var 6) (sNat (b.finalGrad.strides 1)),
       mul key (sNat (b.finalGrad.strides 2)),mul val (sNat (b.finalGrad.strides 3))] else [])


instance : HAdd SExpr SExpr SExpr := ⟨SExpr.add⟩
instance : HSub SExpr SExpr SExpr := ⟨SExpr.sub⟩
instance : HMul SExpr SExpr SExpr := ⟨SExpr.mul⟩
instance (i : Nat) : OfNat SExpr i := ⟨.const i⟩

def formed (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers) : List SExpr :=
  let iv := SExpr.var 0
  let ik := SExpr.var 1
  let h := SExpr.var 2
  let lk := SExpr.var 3
  let lv := SExpr.var 4
  let batch := SExpr.var 5
  let head := SExpr.var 6
  let time := SExpr.var 7
  let key := ik*sNat s.BK+lk
  let val := iv*sNat s.BV+lv
  let q := fun t => sNat b.q.offset+h*sNat (b.q.strides 1)+key+t*sNat s.DK
  let k := fun t => sNat b.k.offset+h*sNat (b.q.strides 1)+key+t*sNat s.DK
  let v := fun t => sNat b.v.offset+h*sNat (b.v.strides 1)+val+t*sNat s.DV
  let do_ := fun t => sNat b.do_.offset+batch*sNat (b.do_.strides 0)+head*sNat (b.do_.strides 1)+
    val*sNat (b.do_.strides 3)+t*sNat (b.do_.strides 2)
  let outV := fun t => (h+ik*sNat s.B*sNat s.H)*sNat (b.v.strides 1)+val+t*sNat s.DV
  let outK := fun t => (h+iv*sNat s.B*sNat s.H)*sNat (b.q.strides 1)+key+t*sNat s.DK
  let state := h*sNat s.DK*sNat s.DV+key*sNat s.DV+val
  let initial := if seed then [sNat b.seed.offset+state] else []
  match phase with
  | .forward => [q time,k time,v time,outV time]++initial++(if extra then [state] else [])
  | .backward =>
      let rev := sNat s.T-1-time
      [q 0,k time,v time,do_ time,outK time,q rev,k rev,v rev,do_ rev,outK rev,outV rev]++initial++
        (if extra then [sNat b.finalGrad.offset+batch*sNat (b.finalGrad.strides 0)+head*sNat (b.finalGrad.strides 1)+
          key*sNat (b.finalGrad.strides 2)+val*sNat (b.finalGrad.strides 3)] else [])

def addressCheck (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers) : Bool :=
  decide (0<s.B ∧ 0<s.H) &&
  ([s.B,s.H,s.T,s.DK,s.DV,s.BK,s.BV,s.NK,s.NV,s.BH,s.NV*s.NK,s.NV*s.NK*s.BH].all
    (fun value => decide (value<2^31))) &&
  (List.range 3).all (fun i => decide (b.q.strides ⟨i%3+1,by omega⟩<2^31 ∧ b.v.strides ⟨i%3+1,by omega⟩<2^31)) &&
  (if phase=.backward then (List.finRange 4).all fun i =>
    decide (b.do_.strides i<2^31 ∧ (extra=Bool.true → b.finalGrad.strides i<2^31)) else Bool.true) &&
  decide (s.NV<2^31 ∧ s.NK≤65535 ∧ s.BH≤65535) &&
  (roots phase seed extra s b).all (SExpr.check (limits s)) &&
  (formed phase seed extra s b).all (fun e => decide (4*e.radius (limits s)<2^63))
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission
