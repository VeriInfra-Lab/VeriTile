import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.Ideal
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.MathLoop
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPForwardInner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPForwardNoninner

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems RealMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

def lift (s : WithBot ℝ × ℝ) : WithBot ℝ × WithBot ℝ := (s.1,(s.2 : WithBot ℝ))

theorem update (m q : WithBot ℝ) (z : ℝ) :
    (FPReference.max Ideal.algebra m q,
      FPReference.add Ideal.algebra
        (FPReference.mul Ideal.algebra (z : WithBot ℝ)
          (FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra m (FPReference.max Ideal.algebra m q))))
        (FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra q (FPReference.max Ideal.algebra m q)))) =
      lift (maskedOnlineStep (m,z) q) := by
  cases m <;> cases q <;> rfl

def innerInput (s : State (WithBot ℝ)) (X : RegionName) (row N : Nat) (n : Nat) : WithBot ℝ :=
  FPReference.read s X (row*N+n)

def noninnerInput (s : State (WithBot ℝ)) (X : RegionName) (row N K k : Nat) (n : Nat) : WithBot ℝ :=
  if k<K then FPReference.read s X (row*N*K+n*K+k) else ⊥

def innerModel (s : State (WithBot ℝ)) (X : RegionName) (row N B c : Nat) :
    FPForwardInner.Acc (WithBot ℝ) B :=
  (fun i => (ExtendedModel.state N B i.1.val c (innerInput s X row N)).1,
   fun i => ((ExtendedModel.state N B i.1.val c (innerInput s X row N)).2 : WithBot ℝ))

def noninnerModel (s : State (WithBot ℝ)) (X : RegionName) (row col N K B C c : Nat) :
    FPForwardNoninner.Acc (WithBot ℝ) B C :=
  (fun i => (ExtendedModel.state N B i.1.val c (noninnerInput s X row N K (col*C+i.2.1.val))).1,
   fun i => ((ExtendedModel.state N B i.1.val c (noninnerInput s X row N K (col*C+i.2.1.val))).2 : WithBot ℝ))

theorem inner_load (s : State (WithBot ℝ)) (X : RegionName) (masked : Bool)
    (row N B c : Nat) (completeTiles : masked=Bool.false → (c+1)*B≤N) (i : TileIndex [B]) :
    FPForwardInner.load Ideal.algebra s X masked row N B (c*B) i =
      if c*B+i.1.val<N then innerInput s X row N (c*B+i.1.val) else ⊥ := by
  have bound := i.1.isLt
  cases masked
  ·
    have h := completeTiles rfl
    have hi : c*B+i.1.val<N := by rw [Nat.add_mul,Nat.one_mul] at h; omega
    simp [FPForwardInner.load,FPForwardInner.active,FPForwardInner.offset,innerInput,Ideal.algebra,hi]
  ·
    simp [FPForwardInner.load,FPForwardInner.active,FPForwardInner.offset,innerInput,Ideal.algebra]

theorem noninner_load (s : State (WithBot ℝ)) (X : RegionName)
    (row col N K B C c : Nat) (i : TileIndex [B,C]) :
    FPForwardNoninner.load Ideal.algebra s X row col N K B C (c*B) i =
      if c*B+i.1.val<N then noninnerInput s X row N K (col*C+i.2.1.val) (c*B+i.1.val) else ⊥ := by
  by_cases hn : c*B+i.1.val<N <;> by_cases hk : col*C+i.2.1.val<K <;>
    simp [FPForwardNoninner.load,FPForwardNoninner.active,FPForwardNoninner.offset,
      noninnerInput,Ideal.algebra,hn,hk]

theorem inner_step (s : State (WithBot ℝ)) (X : RegionName) (masked : Bool)
    (row N B c : Nat) (completeTiles : masked=Bool.false → (c+1)*B≤N) :
    FPForwardInner.accumulate Ideal.algebra s X masked row N B (c*B) (innerModel s X row N B c) =
      innerModel s X row N B (c+1) := by
  apply Prod.ext <;> funext i
  all_goals
    have h := update (ExtendedModel.state N B i.1.val c (innerInput s X row N)).1
      (if c*B+i.1.val<N then innerInput s X row N (c*B+i.1.val) else ⊥)
      (ExtendedModel.state N B i.1.val c (innerInput s X row N)).2
    rw [←ExtendedModel.state_succ] at h
  · simpa only [FPForwardInner.accumulate,innerModel,inner_load s X masked row N B c completeTiles i,lift] using congrArg Prod.fst h
  · simpa only [FPForwardInner.accumulate,innerModel,inner_load s X masked row N B c completeTiles i,lift] using congrArg Prod.snd h

theorem noninner_step (s : State (WithBot ℝ)) (X : RegionName) (row col N K B C c : Nat) :
    FPForwardNoninner.accumulate Ideal.algebra s X row col N K B C (c*B) (noninnerModel s X row col N K B C c) =
      noninnerModel s X row col N K B C (c+1) := by
  apply Prod.ext <;> funext i
  all_goals
    have h := update (ExtendedModel.state N B i.1.val c (noninnerInput s X row N K (col*C+i.2.1.val))).1
      (if c*B+i.1.val<N then noninnerInput s X row N K (col*C+i.2.1.val) (c*B+i.1.val) else ⊥)
      (ExtendedModel.state N B i.1.val c (noninnerInput s X row N K (col*C+i.2.1.val))).2
    rw [←ExtendedModel.state_succ] at h
  · simpa only [FPForwardNoninner.accumulate,noninnerModel,noninner_load,lift] using congrArg Prod.fst h
  · simpa only [FPForwardNoninner.accumulate,noninnerModel,noninner_load,lift] using congrArg Prod.snd h

#axiomsClean inner_step
#axiomsClean noninner_step
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
