import bench.tritonbench_g.mean_reduction.Proofs.WholeNative
import bench.tritonbench_g.mean_reduction.Proofs.WholeFPFamily

namespace VeriTile.Bench.TritonBenchG.MeanReduction.WholeLowered
open VeriTile.Triton FP.Structural WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Only actual native scalar instructions are opaque. There is no reduction,
cast, graph-equality, zero-add rewrite, division-elimination or output callback. -/
structure Scalars where
  add : BitVec 32 → BitVec 32 → BitVec 32
  denominator : Nat → BitVec 32
  divide : BitVec 32 → BitVec 32 → BitVec 32

def sumEight (B : Scalars) (N : Nat) (winner : Fin 8 → Fin 8)
    (acc : Values (BitVec 32) .real [8,8]) : Values (BitVec 32) .real [8] :=
  fun i => if N=1 then orderedOne B.add (fun col => acc (i.1,col,PUnit.unit))
    else orderedDynamic B.add (fun col => acc (i.1,col,PUnit.unit)) (winner i.1)

/-- The exact source reduction site receives its observed ordered native tree.
Other shapes are unused by this kernel and receive an inert interpretation. -/
def sum (B : Scalars) (N : Nat) (winner : Fin 8 → Fin 8) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (acc : Values (BitVec 32) .real shape) :
    Values (BitVec 32) .real (TileShape.reduceShape shape ax keep) := by
  classical
  by_cases hs : shape=[8,8]
  · subst shape
    by_cases ha : ax=⟨1,by simp⟩
    · subst ax
      cases keep
      · exact sumEight B N winner acc
      · exact fun _ => 0
    · exact fun _ => 0
  · exact fun _ => 0

/-- Profile-specific interpretation of the observed source DAG. N=1 executes
native add(load,+0) and returns the reduction directly; no identity or
commutation theorem is assumed. Dynamic N retains add(acc,load) and div.full. -/
def algebra (B : Scalars) (N : Nat) (winner : Fin 8 → Fin 8) : Algebra (BitVec 32) where
  literal := fun _ _ r => if r=0 then 0 else B.denominator N
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => if N=1 then B.add y x else B.add x y
    | .div => if N=1 then x else B.divide x y
    | _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ => B.denominator
  fromInt := fun _ i => B.denominator i.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} => sum B N winner

/-- An arbitrary discrete last-writer selection is supplied for each output.
It carries no payload or numeric result. Different lanes retain different trees;
no equality of replicas or profiles is needed. -/
def family (B : Scalars) (N : Nat) (winner : Nat → Fin 8) (pid : Nat) : Algebra (BitVec 32) :=
  algebra B N (fun row => winner (pid*8+row.val))

def chunk (X : RegionName) (M N pid off : Nat)
    (memory : RegionName → Nat → Cell (BitVec 32)) : Values (BitVec 32) .real [8,8] :=
  fun i => if pid*8+i.1.val<M ∧ off+i.2.1.val<N then
    (memory X ((pid*8+i.1.val)*N+(off+i.2.1.val))).read .real else 0

def next (B : Scalars) (X : RegionName) (M N pid off : Nat)
    (memory : RegionName → Nat → Cell (BitVec 32)) (acc : WholeFPValue.Acc (BitVec 32)) :
    WholeFPValue.Acc (BitVec 32) :=
  fun i => if N=1 then B.add (chunk X M N pid off memory i) (acc i)
    else B.add (acc i) (chunk X M N pid off memory i)

def accumulate (B : Scalars) (X : RegionName) (M N pid : Nat)
    (memory : RegionName → Nat → Cell (BitVec 32)) (off : Nat)
    (acc : WholeFPValue.Acc (BitVec 32)) : WholeFPValue.Acc (BitVec 32) :=
  if h : off<N then accumulate B X M N pid memory (off+8) (next B X M N pid off memory acc)
  else acc
termination_by N-off

theorem accumulated_eq (B : Scalars) (N : Nat) (winner : Fin 8 → Fin 8)
    (X : RegionName) (M pid : Nat) (memory : RegionName → Nat → Cell (BitVec 32))
    (off : Nat) (acc : WholeFPValue.Acc (BitVec 32)) :
    WholeFPValue.accumulated (algebra B N winner) X M N pid memory off acc=
      accumulate B X M N pid memory off acc := by
  conv_lhs => rw [WholeFPValue.accumulated]
  conv_rhs => rw [accumulate]
  split
  · have stepEq : WholeFPValue.nextAcc (algebra B N winner) X M N pid off memory acc=
        next B X M N pid off memory acc := by
      funext i
      simp [WholeFPValue.nextAcc,WholeFPValue.chunk,algebra,next,chunk]
    rw [stepEq]
    exact accumulated_eq B N winner X M pid memory (off+8) _
  · rfl
termination_by N-off

def tileFormula (B : Scalars) (X : RegionName) (M N pid : Nat)
    (memory : RegionName → Nat → Cell (BitVec 32)) (winner : Fin 8 → Fin 8) (lane : Fin 8) : BitVec 32 :=
  let acc := accumulate B X M N pid memory 0 (fun _ => 0)
  if N=1 then orderedOne B.add (fun col => acc (lane,col,PUnit.unit))
  else B.divide (orderedDynamic B.add (fun col => acc (lane,col,PUnit.unit)) (winner lane))
    (B.denominator N)

/-- Definitional graph connection: the complete source primitive formula is
exactly the native ordered graph. No numerical hypothesis occurs. -/
theorem tileFormula_eq (B : Scalars) (N : Nat) (positive : 0<N)
    (winner : Fin 8 → Fin 8) (X : RegionName) (M pid : Nat)
    (memory : RegionName → Nat → Cell (BitVec 32)) (lane : Fin 8) :
    WholeFPValue.tileValue (algebra B N winner) X M N pid memory lane=
      tileFormula B X M N pid memory winner lane := by
  have hn : (N:ℝ)≠0 := by exact_mod_cast Nat.ne_of_gt positive
  unfold WholeFPValue.tileValue
  have hz : (algebra B N winner).literal (some .fp32) .real 0=0 := by simp [algebra]
  rw [hz,accumulated_eq]
  by_cases h : N=1
  · simp [WholeFPValue.meanValues,algebra,sum,sumEight,tileFormula,h]
  · simp [WholeFPValue.meanValues,algebra,sum,sumEight,tileFormula,h,hn]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.WholeLowered
