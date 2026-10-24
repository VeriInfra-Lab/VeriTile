import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  DK : Nat
  DV : Nat
  BK : Nat
  BV : Nat

structure Tile where
  bh : Nat
  chunk : Nat
  key : Nat
  value : Nat

/-- A compiler-selected finite SIMT plan, indexed only by source site and
integer coordinates. Its payload is syntax, never a numerical function. -/
structure Plans where
  tree : List Nat → Nat → DotPlan
  complete : ∀ site n,(tree site n).Covers n

def Plans.linear : Plans := ⟨fun _ n => DotPlan.linear n,fun _ n => DotPlan.linear_complete n⟩
def dot (a : Ops α) (p : Plans) (site : List Nat) (n : Nat)
    (x y : Nat → α) (carry : α) := (p.tree site n).eval a x y carry

structure Inputs (α : Type) where
  q : Nat → Nat → Nat → α
  k : Nat → Nat → Nat → α
  v : Nat → Nat → Nat → α
  g : Nat → Nat → Nat → α

/-- Every source block load has explicit zero padding in both coordinates. -/
def load (a : Ops α) (T D : Nat) (x : Nat → Nat → Nat → α) (bh t e : Nat) : α :=
  if t<T ∧ e<D then x bh t e else a.zero

def diagonalDecay (a : Ops α) (b : α) (row col : Nat) : α :=
  let exponent := decayInt a b ((row : Int)-(col : Int))
  if col≤row then exponent else a.zero

/-- Forward at one output lane; the outer dot carries the actual accumulator,
as in the retained TTIR. No reassociation or FP=Real rule is used. -/
def forwardStep (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r v : Nat) (diagonal : Bool) (j : Nat) (carry : α) : α :=
  let b := logDecay a (tile.bh%s.H)
  let t := tile.chunk*128+r
  let start := (if diagonal then tile.chunk*128 else 0)+j*32
  let q := fun e => a.cast (a.mul (load a s.T s.DK x.q tile.bh t (tile.key*s.BK+e)) (a.scale s.DK))
  let score := fun u =>
    let raw := dot a p [0,0,tile.bh,tile.chunk,tile.key,tile.value,r,j,u] s.BK q
      (fun e => load a s.T s.DK x.k tile.bh (start+u) (tile.key*s.BK+e)) a.zero
    a.cast (a.mul raw (if diagonal then diagonalDecay a b r (j*32+u) else decayNat a b (32-u)))
  dot a p [0,1,tile.bh,tile.chunk,tile.key,tile.value,r,j,v] 32 score
    (fun u => load a s.T s.DV x.v tile.bh (start+u) (tile.value*s.BV+v))
    (if diagonal then carry else a.mul carry (blockDecay a b 32))

def forwardLane (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r v : Nat) : α :=
  let before := foldN (forwardStep a p s x tile r v false) (tile.chunk*4) a.zero
  let between := a.mul before (decayNat a (logDecay a (tile.bh%s.H)) r)
  a.cast (foldN (forwardStep a p s x tile r v true) 4 between)

def queryStep (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r e : Nat) (diagonal : Bool) (j : Nat) (carry : α) : α :=
  let b := logDecay a (tile.bh%s.H)
  let t := tile.chunk*64+r
  let start := (if diagonal then tile.chunk*64 else 0)+j*32
  let score := fun u =>
    let raw := dot a p [1,0,tile.bh,tile.chunk,tile.key,tile.value,r,j,u] s.BV
      (fun v => load a s.T s.DV x.g tile.bh t (tile.value*s.BV+v))
      (fun v => load a s.T s.DV x.v tile.bh (start+u) (tile.value*s.BV+v)) a.zero
    a.cast (if diagonal then a.mul (a.mul raw (diagonalDecay a b r (j*32+u))) (a.scale s.DK)
      else a.mul raw (decayNat a b (32-u)))
  dot a p [1,1,tile.bh,tile.chunk,tile.key,tile.value,r,j,e] 32 score
    (fun u => load a s.T s.DK x.k tile.bh (start+u) (tile.key*s.BK+e))
    (if diagonal then carry else a.mul carry (blockDecay a b 32))

def queryLane (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r e : Nat) : α :=
  let before := foldN (queryStep a p s x tile r e false) (tile.chunk*2) a.zero
  let between := a.mul before (a.mul (decayNat a (logDecay a (tile.bh%s.H)) r) (a.scale s.DK))
  a.cast (foldN (queryStep a p s x tile r e true) 2 between)

/-- DK and DV share source loads, but retain their distinct scale placement. -/
def keyValueFuture (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r e v j : Nat) (carry : α×α) : α×α :=
  let b := logDecay a (tile.bh%s.H)
  let n := tile.chunk*64+r
  let start := (blocks s.T 32-1-j)*32
  let q := fun f u => load a s.T s.DK x.q tile.bh (start+u) (tile.key*s.BK+f)
  let g := fun f u => a.cast (a.mul (load a s.T s.DV x.g tile.bh (start+u) (tile.value*s.BV+f)) (decayNat a b u))
  let ks := fun u => a.cast (dot a p [2,0,tile.bh,tile.chunk,tile.key,tile.value,r,j,u] s.BV
    (fun f => load a s.T s.DV x.v tile.bh n (tile.value*s.BV+f)) (fun f => g f u) a.zero)
  let vs := fun u => a.cast (dot a p [2,1,tile.bh,tile.chunk,tile.key,tile.value,r,j,u] s.BK
    (fun f => a.cast (a.mul (load a s.T s.DK x.k tile.bh n (tile.key*s.BK+f)) (decayNat a b (64-r))))
    (fun f => q f u) a.zero)
  (dot a p [2,2,tile.bh,tile.chunk,tile.key,tile.value,r,j,e] 32 ks (fun u => q e u)
     (a.mul carry.1 (blockDecay a b 32)),
   dot a p [2,3,tile.bh,tile.chunk,tile.key,tile.value,r,j,v] 32 vs (fun u => g v u)
     (a.mul carry.2 (blockDecay a b 32)))

def keyValueDiagonal (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r e v j : Nat) (carry : α×α) : α×α :=
  let b := logDecay a (tile.bh%s.H)
  let n := tile.chunk*64+r
  let start := tile.chunk*64+j*32
  let q := fun f u => load a s.T s.DK x.q tile.bh (start+u) (tile.key*s.BK+f)
  let g := fun f u => load a s.T s.DV x.g tile.bh (start+u) (tile.value*s.BV+f)
  let decay := fun u => a.mul (diagonalDecay a (a.cast b) (j*32+u) r) (a.scale s.DK)
  let ks := fun u => a.cast (a.mul (dot a p [3,0,tile.bh,tile.chunk,tile.key,tile.value,r,j,u] s.BV
    (fun f => load a s.T s.DV x.v tile.bh n (tile.value*s.BV+f)) (fun f => g f u) a.zero) (decay u))
  let vs := fun u => a.cast (a.mul (dot a p [3,1,tile.bh,tile.chunk,tile.key,tile.value,r,j,u] s.BK
    (fun f => load a s.T s.DK x.k tile.bh n (tile.key*s.BK+f)) (fun f => q f u) a.zero) (decay u))
  (dot a p [3,2,tile.bh,tile.chunk,tile.key,tile.value,r,j,e] 32 ks (fun u => q e u) carry.1,
   dot a p [3,3,tile.bh,tile.chunk,tile.key,tile.value,r,j,v] 32 vs (fun u => g v u) carry.2)

def keyValueLane (a : Ops α) (p : Plans) (s : Shape) (x : Inputs α)
    (tile : Tile) (r e v : Nat) : α×α :=
  let before := foldN (keyValueFuture a p s x tile r e v) (blocks s.T 32-(tile.chunk+1)*2) (a.zero,a.zero)
  let between := (a.mul before.1 (a.mul (decayNat a (logDecay a (tile.bh%s.H)) (64-r)) (a.scale s.DK)),
    a.mul before.2 (a.scale s.DK))
  let after := foldN (keyValueDiagonal a p s x tile r e v) 2 between
  (a.cast after.1,a.cast after.2)

/-- A pointwise backend correspondence contains only scalar leaves.
The dot plan, loop shape, loads, and stores cannot be replaced by callbacks. -/
structure ScalarAgreement (a b : Ops α) : Prop where
  zero : a.zero=b.zero
  one : a.one=b.one
  negFive : a.negFive=b.negFive
  fromNat : ∀ x,a.fromNat x=b.fromNat x
  fromInt : ∀ x,a.fromInt x=b.fromInt x
  add : ∀ x y,a.add x y=b.add x y
  sub : ∀ x y,a.sub x y=b.sub x y
  mul : ∀ x y,a.mul x y=b.mul x y
  fma : ∀ x y z,a.fma x y z=b.fma x y z
  exp2 : ∀ x,a.exp2 x=b.exp2 x
  log2 : ∀ x,a.log2 x=b.log2 x
  cast : ∀ x,a.cast x=b.cast x
  scale : ∀ n,a.scale n=b.scale n

theorem ScalarAgreement.eq (a b : Ops α) (h : ScalarAgreement a b) : a=b := by
  cases a; cases b
  rcases h with ⟨h0,h1,h5,hn,hi,ha,hs,hm,hf,he,hl,hc,hscale⟩
  simp only [Ops.mk.injEq]
  exact ⟨h0,h1,h5,funext hn,funext hi,funext (fun x => funext (ha x)),
    funext (fun x => funext (hs x)),funext (fun x => funext (hm x)),
    funext (fun x => funext (fun y => funext (hf x y))),funext he,funext hl,funext hc,funext hscale⟩

theorem graph_backend (a b : Ops α) (h : ScalarAgreement a b) (p : Plans) (s : Shape)
    (x : Inputs α) (tile : Tile) (r e v : Nat) :
    forwardLane a p s x tile r v=forwardLane b p s x tile r v ∧
    queryLane a p s x tile r e=queryLane b p s x tile r e ∧
    keyValueLane a p s x tile r e v=keyValueLane b p s x tile r e v := by
  rw [h.eq]
  exact ⟨rfl,rfl,rfl⟩

#axiomsClean ScalarAgreement.eq
#axiomsClean graph_backend
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
