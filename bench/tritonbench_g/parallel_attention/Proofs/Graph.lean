import bench.tritonbench_g.parallel_attention.Proofs.Arithmetic
import bench.tritonbench_g.parallel_attention.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat
  useScale : Bool

structure Tile where
  bh : Nat
  chunk : Nat
  key : Nat
  value : Nat

/-- Finite scalar syntax for actual dot/sum arithmetic, never a tensor callback. -/
structure Plans where
  dotTree : List Nat → Nat → DotPlan
  dotComplete : ∀ site n, (dotTree site n).Covers n
  sumTree : List Nat → Nat → SumPlan
  sumComplete : ∀ site n, (sumTree site n).Covers n

def dot (a : Ops α) (p : Plans) (site : List Nat) (n : Nat) (x y : Nat → α) : α :=
  (p.dotTree site n).eval a x y a.zero

def total (a : Ops α) (p : Plans) (site : List Nat) (n : Nat) (x : Nat → α) : α :=
  (p.sumTree site n).eval a x

structure TensorInputs (α : Type) where
  q : Nat → Nat → Nat → α
  k : Nat → Nat → Nat → α
  v : Nat → Nat → Nat → α
  do_ : Nat → Nat → Nat → α
  dz : Nat → Nat → α

/-- The repaired source specifies zero padding at every block/masked load. -/
def load (a : Ops α) (T D : Nat) (x : Nat → Nat → Nat → α) (bh t e : Nat) : α :=
  if t<T ∧ e<D then x bh t e else a.zero

def dzLoad (a : Ops α) (s : Shape) (x : TensorInputs α) (bh t : Nat) : α :=
  if t<s.T then x.dz bh t else a.zero

def queryLoad (a : Ops α) (s : Shape) (x : TensorInputs α) (tile : Tile) (L r e : Nat) : α :=
  a.cast (a.mul (load a s.T s.K x.q tile.bh (tile.chunk*L+r) (tile.key*s.BK+e))
    (a.scale s.K s.useScale))

/-- Both forward loops retain the original square/mask/cast/add order. -/
def forwardStep (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r v : Nat) (diagonal : Bool) (j : Nat) (carry : α×α) : α×α :=
  let start := (if diagonal then tile.chunk*128 else 0)+j*32
  let scores := fun u =>
    let raw := dot a p [0,0,tile.bh,tile.chunk,tile.value,r,j,u] s.BK
      (queryLoad a s x tile 128 r)
      (fun e => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+e))
    let squared := a.mul raw raw
    if diagonal && decide (r < j*32+u) then a.zero else squared
  let norm := total a p [0,1,tile.bh,tile.chunk,tile.value,r,j] 32 scores
  let output := dot a p [0,2,tile.bh,tile.chunk,tile.value,r,j,v] 32
    (fun u => a.cast (scores u))
    (fun u => load a s.T s.V x.v tile.bh (start+u) (tile.value*s.BV+v))
  (a.add carry.1 output, a.add carry.2 norm)

def forwardLane (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r v : Nat) : α×α :=
  let prior := foldN (forwardStep a p s x tile r v false) (tile.chunk*4) (a.zero,a.zero)
  let result := foldN (forwardStep a p s x tile r v true) 4 prior
  (a.cast result.1, a.cast result.2)

/-- DQ's prefix is scaled after its loop; the diagonal scales each DS first. -/
def queryStep (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e : Nat) (diagonal : Bool) (j : Nat) (carry : α) : α :=
  let t := tile.chunk*64+r
  let start := (if diagonal then tile.chunk*64 else 0)+j*32
  let terms := fun u =>
    let ds0 := dot a p [1,0,tile.bh,tile.chunk,tile.value,r,j,u] s.BV
      (fun v => a.cast (load a s.T s.V x.do_ tile.bh t (tile.value*s.BV+v)))
      (fun v => load a s.T s.V x.v tile.bh (start+u) (tile.value*s.BV+v))
    let ds1 := if tile.value=0 then a.add ds0 (dzLoad a s x tile.bh t) else ds0
    let raw := dot a p [1,1,tile.bh,tile.chunk,tile.value,r,j,u] s.BK
      (queryLoad a s x tile 64 r)
      (fun k => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+k))
    let masked := diagonal && decide (r < j*32+u)
    let ds := if diagonal then a.mul (if masked then a.zero else ds1) (a.scale s.K s.useScale) else ds1
    let score := if masked then a.zero else raw
    a.cast (a.mul (a.mul a.two ds) score)
  a.add carry (dot a p [1,2,tile.bh,tile.chunk,tile.value,r,j,e] 32 terms
    (fun u => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+e)))

def queryLane (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e : Nat) : α :=
  let prior := foldN (queryStep a p s x tile r e false) (tile.chunk*2) a.zero
  let scaled := a.mul prior (a.scale s.K s.useScale)
  a.cast (foldN (queryStep a p s x tile r e true) 2 scaled)

/-- One backward DK/DV source iteration; future and diagonal orders differ. -/
def keyValueStep (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e v : Nat) (diagonal : Bool) (j : Nat) (carry : α×α) : α×α :=
  let n := tile.chunk*64+r
  let start := if diagonal then tile.chunk*64+j*32 else (blocks s.T 32-1-j)*32
  let q := fun f u => load a s.T s.K x.q tile.bh (start+u) (tile.key*s.BK+f)
  let do_ := fun f u => a.cast (load a s.T s.V x.do_ tile.bh (start+u) (tile.value*s.BV+f))
  let valid := fun u => decide (start+u<s.T) && (!diagonal || decide (r≤j*32+u))
  let rawScore := fun u => a.mul (dot a p [2,0,tile.bh,tile.chunk,tile.value,r,j,u] s.BK
    (fun f => let k := load a s.T s.K x.k tile.bh n (tile.key*s.BK+f)
              if diagonal then k else a.cast k) (fun f => q f u)) (a.scale s.K s.useScale)
  let score := fun u => if valid u then rawScore u else a.zero
  let square := fun u => if diagonal then
      if valid u then a.mul (rawScore u) (rawScore u) else a.zero
    else a.mul (score u) (score u)
  let ds := fun u =>
    let raw := dot a p [2,1,tile.bh,tile.chunk,tile.value,r,j,u] s.BV
      (fun f => load a s.T s.V x.v tile.bh n (tile.value*s.BV+f)) (fun f => do_ f u)
    let z := dzLoad a s x tile.bh (start+u)
    if diagonal then
      a.mul (if valid u then (if tile.value=0 then a.add raw z else raw) else a.zero)
        (a.scale s.K s.useScale)
    else
      if valid u then
        if tile.value=0 then a.add (a.mul raw (a.scale s.K s.useScale)) (a.mul z (a.scale s.K s.useScale))
        else a.mul raw (a.scale s.K s.useScale)
      else a.zero
  let dv := dot a p [2,2,tile.bh,tile.chunk,tile.value,r,j,v] 32
    (fun u => a.cast (square u)) (fun u => do_ v u)
  let dk := dot a p [2,3,tile.bh,tile.chunk,tile.value,r,j,e] 32
    (fun u => a.cast (a.mul (a.mul a.two (ds u)) (score u))) (fun u => q e u)
  (a.add carry.1 dk, a.add carry.2 dv)

def keyValueLane (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e v : Nat) : α×α :=
  let future := foldN (keyValueStep a p s x tile r e v false)
    (blocks s.T 32-(tile.chunk+1)*2) (a.zero,a.zero)
  let result := foldN (keyValueStep a p s x tile r e v true) 2 future
  (a.cast result.1,a.cast result.2)

/-- No arithmetic deferral may replace loops, tensors, outputs or memory. -/
structure ScalarAgreement (a b : Ops α) : Prop where
  zero : a.zero=b.zero
  one : a.one=b.one
  two : a.two=b.two
  add : ∀ x y,a.add x y=b.add x y
  mul : ∀ x y,a.mul x y=b.mul x y
  fma : ∀ x y z,a.fma x y z=b.fma x y z
  div : ∀ x y,a.div x y=b.div x y
  neg : ∀ x,a.neg x=b.neg x
  cast : ∀ x,a.cast x=b.cast x
  scale : ∀ n flag,a.scale n flag=b.scale n flag

theorem ScalarAgreement.eq (a b : Ops α) (h : ScalarAgreement a b) : a=b := by
  cases a; cases b
  rcases h with ⟨hz,ho,ht,ha,hm,hf,hd,hn,hc,hs⟩
  simp only [Ops.mk.injEq]
  exact ⟨hz,ho,ht,funext (fun x => funext (ha x)),funext (fun x => funext (hm x)),
    funext (fun x => funext (fun y => funext (hf x y))),funext (fun x => funext (hd x)),
    funext hn,funext hc,funext (fun n => funext (hs n))⟩

theorem graph_backend (a b : Ops α) (same : ScalarAgreement a b) (p : Plans) (s : Shape)
    (x : TensorInputs α) (tile : Tile) (r e v : Nat) :
    forwardLane a p s x tile r v=forwardLane b p s x tile r v ∧
    queryLane a p s x tile r e=queryLane b p s x tile r e ∧
    keyValueLane a p s x tile r e v=keyValueLane b p s x tile r e v := by
  rw [same.eq]
  exact ⟨rfl,rfl,rfl⟩

#axiomsClean ScalarAgreement.eq
#axiomsClean graph_backend
end
end VeriTile.Bench.Repairs.ParallelAttention
