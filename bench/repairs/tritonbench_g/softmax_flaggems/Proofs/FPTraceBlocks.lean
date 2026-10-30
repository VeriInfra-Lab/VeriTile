import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPGrid
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPTraceCore

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTrace
open VeriTile Triton FP.Structural FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

def backwardInnerTile (M : Algebra α) (P G Y : RegionName) (row rows N B C : Nat)
    (scale : Values α .real [C]) (cur : Nat) : Tile α :=
  ⟨[C,B],Y,FPBackwardInner.offset row N cur,FPBackwardInner.active row rows N cur,
    fun s => FPBackwardInner.storedValue M s P G row rows N B C cur scale⟩

def backwardNoninnerTile (M : Algebra α) (P G Y : RegionName) (row col rows N K B C : Nat)
    (scale : Values α .real [C]) (cur : Nat) : Tile α :=
  ⟨[B,C],Y,FPBackwardNoninner.offset row col N K cur,FPBackwardNoninner.active row col rows N K cur,
    fun s => FPBackwardNoninner.storedValue M s P G row col rows N K B C cur scale⟩

def forwardNoninnerTile (M : Algebra α) (X Y : RegionName) (row col N K B C : Nat)
    (mx z : Values α .real [C]) (cur : Nat) : Tile α :=
  ⟨[B,C],Y,FPForwardNoninner.offset row col N K (FPForwardNoninner.previous N B-cur),
    FPForwardNoninner.active col N K (FPForwardNoninner.previous N B-cur),
    fun s => FPForwardNoninner.storedValue M s X row col N K B C cur mx z⟩

def forwardInnerTile (M : Algebra α) (X Y : RegionName) (masked : Bool) (row N B : Nat)
    (mx z : α) (cur : Nat) : Tile α :=
  ⟨[B],Y,FPForwardInner.offset row N (FPForwardInner.previous N B-cur),
    FPForwardInner.active masked N (FPForwardInner.previous N B-cur),
    fun s => FPForwardInner.storedValue M s X masked row N B cur mx z⟩

def backwardTiles (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (base : State α) : List (Tile α) :=
  if 1<d.K then
    if c.one then [⟨[c.B,c.C],Y,noninnerOffset row col d.N d.K,noninnerActive col d.N d.K,
      fun s => backwardNoninnerValue M s P G row col d.N d.K c.B c.C⟩]
    else
      let v := (walk (FPBackwardNoninner.accumulate M base P G row col d.M d.N d.K c.B c.C)
        d.N c.B 0 (fun _ => literal M 0)).2
      (visits d.N c.B 0).map (backwardNoninnerTile M P G Y row col d.M d.N d.K c.B c.C
        (FPBackwardNoninner.reduced M c.B c.C v))
  else
    if c.one then [⟨[c.C,c.B],Y,backwardInnerOffset row d.N,backwardInnerActive row d.M d.N,
      fun s => backwardInnerValue M s P G row d.M d.N c.B c.C⟩]
    else
      let v := (walk (FPBackwardInner.accumulate M base P G row d.M d.N c.B c.C)
        d.N c.B 0 (fun _ => literal M 0)).2
      (visits d.N c.B 0).map (backwardInnerTile M P G Y row d.M d.N c.B c.C
        (FPBackwardInner.reduced M c.B c.C v))

def forwardTiles (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (base : State α) : List (Tile α) :=
  if 1<d.K then
    if c.one then [⟨[c.B,c.C],Y,noninnerOffset row col d.N d.K,noninnerActive col d.N d.K,
      fun s => forwardNoninnerValue M s X row col d.N d.K c.B c.C⟩]
    else
      let v := (walk (FPForwardNoninner.accumulate M base X row col d.N d.K c.B c.C)
        d.N c.B 0 ((fun _ => M.negInf),fun _ => literal M 0)).2
      (visits d.N c.B 0).map (forwardNoninnerTile M X Y row col d.N d.K c.B c.C
        (FPForwardNoninner.maxReduced M c.B c.C v) (FPForwardNoninner.sumReduced M c.B c.C v))
  else
    if c.one then [⟨[c.B],Y,innerOffset row d.N,innerActive d.N,
      fun s => forwardInnerValue M s X row d.N c.B⟩]
    else
      let first := (walk (FPForwardInner.accumulate M base X Bool.false row d.N c.B)
        (FPForwardInner.previous d.N c.B) c.B 0 ((fun _ => M.negInf),fun _ => literal M 0)).2
      let v := (walk (FPForwardInner.accumulate M base X Bool.true row d.N c.B)
        d.N c.B (FPForwardInner.previous d.N c.B) first).2
      let mx := FPForwardInner.maxReduced M c.B v
      let z := FPForwardInner.sumReduced M c.B v
      ((visits c.B c.B 0).map (forwardInnerTile M X Y Bool.true row d.N c.B mx z)) ++
      ((visits d.N c.B c.B).map (forwardInnerTile M X Y Bool.false row d.N c.B mx z))

theorem backward_eq (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (s : State α) :
    FPGrid.backwardBlock M P G Y d c row col s = execute (backwardTiles M P G Y d c row col s) s := by
  unfold FPGrid.backwardBlock backwardTiles
  split <;> cases hc : c.one
  · dsimp [FPBackwardNoninner.graph]
    exact walk_tiles (backwardNoninnerTile M P G Y row col d.M d.N d.K c.B c.C _) d.N c.B 0 s
  · rfl
  · dsimp [FPBackwardInner.graph]
    exact walk_tiles (backwardInnerTile M P G Y row d.M d.N c.B c.C _) d.N c.B 0 s
  · rfl

theorem forward_eq (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (s : State α) :
    FPGrid.forwardBlock M X Y d c row col s = execute (forwardTiles M X Y d c row col s) s := by
  unfold FPGrid.forwardBlock forwardTiles
  split <;> cases hc : c.one
  · dsimp [FPForwardNoninner.graph]
    exact walk_tiles (forwardNoninnerTile M X Y row col d.N d.K c.B c.C _ _) d.N c.B 0 s
  · rfl
  · dsimp [FPForwardInner.graph]
    rw [execute_append,←walk_tiles,←walk_tiles]
    rfl
  · rfl

#axiomsClean backward_eq
#axiomsClean forward_eq
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTrace
