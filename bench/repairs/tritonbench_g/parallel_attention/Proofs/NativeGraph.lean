import bench.repairs.tritonbench_g.parallel_attention.Proofs.Lowering

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

def nativeQueryLoad (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (L r e : Nat) : α :=
  a.cast (nativeScale a c s [3,0,tile.bh,tile.chunk,tile.value,r,e]
    (load a s.T s.K x.q tile.bh (tile.chunk*L+r) (tile.key*s.BK+e)))

/-- TTIR combines the source O addition into the live dot accumulator; Z
retains its separate reduction and addition. -/
def nativeForwardStep (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r v : Nat) (diagonal : Bool) (j : Nat) (carry : α×α) : α×α :=
  let start := (if diagonal then tile.chunk*128 else 0)+j*32
  let raw := fun u => dot a c.plans [0,0,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] s.BK
    (nativeQueryLoad a c s x tile 128 r)
    (fun e => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+e))
  let masked := fun u => diagonal && decide (r < j*32+u)
  let scores := fun u => if masked u then a.zero else a.mul (raw u) (raw u)
  let norm := nativeNorm a c [0,1,tile.bh,tile.chunk,tile.value,r,j,diagonal.toNat] 32 raw masked
  let output := liveDot a c.plans [0,2,tile.bh,tile.chunk,tile.value,r,j,v,diagonal.toNat] 32
    (fun u => a.cast (scores u))
    (fun u => load a s.T s.V x.v tile.bh (start+u) (tile.value*s.BV+v)) carry.1
  (output,a.add carry.2 norm)

def nativeForwardLane (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r v : Nat) : α×α :=
  let prior := foldN (nativeForwardStep a c s x tile r v false) (tile.chunk*4) (a.zero,a.zero)
  let result := foldN (nativeForwardStep a c s x tile r v true) 4 prior
  (a.cast result.1,a.cast result.2)

/-- DS is seeded with dZ on the unique value shard. Diagonal scale/doubling
uses explicit selected scalar instruction syntax, including compiler FMAs. -/
def nativeQueryStep (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e : Nat) (diagonal : Bool) (j : Nat) (carry : α) : α :=
  let t := tile.chunk*64+r
  let start := (if diagonal then tile.chunk*64 else 0)+j*32
  let terms := fun u =>
    let ds := liveDot a c.plans [1,0,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] s.BV
      (fun v => a.cast (load a s.T s.V x.do_ tile.bh t (tile.value*s.BV+v)))
      (fun v => load a s.T s.V x.v tile.bh (start+u) (tile.value*s.BV+v))
      (if tile.value=0 then dzLoad a s x tile.bh t else a.zero)
    let raw := dot a c.plans [1,1,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] s.BK
      (nativeQueryLoad a c s x tile 64 r)
      (fun k => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+k))
    let masked := diagonal && decide (r < j*32+u)
    let twice := if diagonal then nativeScaleTwice a c s [3,1,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
      else nativeTwice a c [3,2,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
    a.cast (nativeCoefficient a c [3,9,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
      (!masked) twice ds raw)
  liveDot a c.plans [1,2,tile.bh,tile.chunk,tile.value,r,j,e,diagonal.toNat] 32 terms
    (fun u => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+e)) carry

def nativeQueryLane (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e : Nat) : α :=
  let prior := foldN (nativeQueryStep a c s x tile r e false) (tile.chunk*2) a.zero
  let scaled := nativeScale a c s [3,3,tile.bh,tile.chunk,tile.value,r,e] prior
  a.cast (foldN (nativeQueryStep a c s x tile r e true) 2 scaled)

/-- Scaled future DS has a zero dot seed and an explicit contracted scalar
sum. No-scale future DS absorbs dZ into its dot seed. Diagonal DS always uses
the dZ seed. DK and DV dots retain their previous live accumulators. -/
def nativeKeyValueStep (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e v : Nat) (diagonal : Bool) (j : Nat) (carry : α×α) : α×α :=
  let n := tile.chunk*64+r
  let start := if diagonal then tile.chunk*64+j*32 else (blocks s.T 32-1-j)*32
  let q := fun f u => load a s.T s.K x.q tile.bh (start+u) (tile.key*s.BK+f)
  let do_ := fun f u => a.cast (load a s.T s.V x.do_ tile.bh (start+u) (tile.value*s.BV+f))
  let valid := fun u => decide (start+u<s.T) && (!diagonal || decide (r≤j*32+u))
  let rawScore := fun u => nativeScale a c s [3,4,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
    (dot a c.plans [2,0,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] s.BK
      (fun f => let k := load a s.T s.K x.k tile.bh n (tile.key*s.BK+f)
                if diagonal then k else a.cast k) (fun f => q f u))
  let sc := fun u => if valid u then rawScore u else a.zero
  let square := fun u => if diagonal then
      if valid u then a.mul (rawScore u) (rawScore u) else a.zero
    else a.mul (sc u) (sc u)
  let ds := fun u =>
    let z := dzLoad a s x tile.bh (start+u)
    let seed := if tile.value=0 ∧ (diagonal=true ∨ s.useScale=false) then z else a.zero
    let raw := liveDot a c.plans [2,1,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] s.BV
      (fun f => load a s.T s.V x.v tile.bh n (tile.value*s.BV+f)) (fun f => do_ f u) seed
    if diagonal then raw else if s.useScale then
      if tile.value=0 then nativeScaledSum a c s [3,6,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] raw z
      else nativeScale a c s [3,7,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat] raw
    else raw
  let twice := fun u => if diagonal then nativeScaleTwice a c s [3,5,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
    else nativeTwice a c [3,8,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
  let dv := liveDot a c.plans [2,2,tile.bh,tile.chunk,tile.value,r,j,v,diagonal.toNat] 32
    (fun u => a.cast (square u)) (fun u => do_ v u) carry.2
  let dk := liveDot a c.plans [2,3,tile.bh,tile.chunk,tile.value,r,j,e,diagonal.toNat] 32
    (fun u => a.cast (nativeCoefficient a c [3,10,tile.bh,tile.chunk,tile.value,r,j,u,diagonal.toNat]
      (valid u) (twice u) (ds u) (rawScore u))) (fun u => q e u) carry.1
  (dk,dv)

def nativeKeyValueLane (a : Ops α) (c : CompilerLayout) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r e v : Nat) : α×α :=
  let future := foldN (nativeKeyValueStep a c s x tile r e v false)
    (blocks s.T 32-(tile.chunk+1)*2) (a.zero,a.zero)
  let result := foldN (nativeKeyValueStep a c s x tile r e v true) 2 future
  (a.cast result.1,a.cast result.2)

end
end VeriTile.Bench.Repairs.ParallelAttention
