import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealForwardBlock

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

def Read (R : RegionName) (v : Nat → ℝ) (count : Nat) (s : State (WithBot ℝ)) : Prop :=
  ∀ o, o<count → FPReference.read s R o=(v o : WithBot ℝ)
def Input (P G : RegionName) (p g : Nat → ℝ) (count : Nat) (s : State (WithBot ℝ)) :=
  Read P p count s ∧ Read G g count s

def lane (N B c l : Nat) (v : Nat → ℝ) : ℝ :=
  ∑ t ∈ Finset.range c, if t*B+l<N then v (t*B+l) else 0

def innerModel (p g : Nat → ℝ) (row rows N B C c : Nat) : Values (WithBot ℝ) .real [C,B] :=
  fun i => (lane N B c i.2.1.val (fun n => if row*C+i.1.val<rows then
    p ((row*C+i.1.val)*N+n)*g ((row*C+i.1.val)*N+n) else 0) : WithBot ℝ)
def noninnerModel (p g : Nat → ℝ) (row col N K B C c : Nat) : Values (WithBot ℝ) .real [B,C] :=
  fun i => (lane N B c i.1.val (fun n => if col*C+i.2.1.val<K then
    p (row*N*K+n*K+(col*C+i.2.1.val))*g (row*N*K+n*K+(col*C+i.2.1.val)) else 0) : WithBot ℝ)

@[simp] theorem lane_zero (N B l : Nat) (v : Nat → ℝ) : lane N B 0 l v=0 := by simp [lane]
theorem lane_succ (N B c l : Nat) (v : Nat → ℝ) :
    lane N B (c+1) l v=lane N B c l v+(if c*B+l<N then v (c*B+l) else 0) := by
  exact Finset.sum_range_succ _ _

theorem lane_total (N B : Nat) (hB : 0<B) (v : Nat → ℝ) :
    (∑ l : Fin B, lane N B (RealAddress.rounds N B) l.val v)=∑ n : Fin N, v n.val := by
  simpa only [lane,Fin.sum_univ_eq_sum_range] using
    RealMath.lane_tile_partition N B hB (RealAddress.rounds N B) (RealAddress.coverage hB) v

theorem inner_load (s : State (WithBot ℝ)) (R : RegionName) (v : Nat → ℝ)
    (row rows N B C cur : Nat) (read : Read R v (rows*N) s) (i : TileIndex [C,B]) :
    FPBackwardInner.load Ideal.algebra s R row rows N B C cur i=
      ((if row*C+i.1.val<rows ∧ cur+i.2.1.val<N then v ((row*C+i.1.val)*N+(cur+i.2.1.val)) else 0 : ℝ) : WithBot ℝ) := by
  by_cases active : row*C+i.1.val<rows ∧ cur+i.2.1.val<N
  · have bound := inner_bound active.1 active.2
    simp [FPBackwardInner.load,FPBackwardInner.active,FPBackwardInner.offset,Ideal.algebra,
      active.1,active.2,read _ bound,active]
  · simp only [FPBackwardInner.load,FPBackwardInner.active,FPBackwardInner.offset,
      Bool.and_eq_true,decide_eq_true_eq,if_neg active,Ideal.literal_value]
    rfl

theorem noninner_load (s : State (WithBot ℝ)) (R : RegionName) (v : Nat → ℝ)
    (row col rows N K B C cur : Nat) (hr : row<rows) (read : Read R v (rows*N*K) s)
    (i : TileIndex [B,C]) :
    FPBackwardNoninner.load Ideal.algebra s R row col rows N K B C cur i=
      ((if cur+i.1.val<N ∧ col*C+i.2.1.val<K then
        v (row*N*K+(cur+i.1.val)*K+(col*C+i.2.1.val)) else 0 : ℝ) : WithBot ℝ) := by
  by_cases active : cur+i.1.val<N ∧ col*C+i.2.1.val<K
  · have bound := RealAddress.non_inner_bound hr active.1 active.2
    simp [FPBackwardNoninner.load,FPBackwardNoninner.active,FPBackwardNoninner.offset,Ideal.algebra,
      active.1,active.2,read _ bound,active]
  · simp only [FPBackwardNoninner.load,FPBackwardNoninner.active,FPBackwardNoninner.offset,
      Bool.and_eq_true,decide_eq_true_eq,if_neg active,Ideal.literal_value]
    rfl

theorem inner_step (s : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row rows N B C c : Nat) (input : Input P G p g (rows*N) s) :
    FPBackwardInner.accumulate Ideal.algebra s P G row rows N B C (c*B)
      (innerModel p g row rows N B C c)=innerModel p g row rows N B C (c+1) := by
  funext i
  simp only [FPBackwardInner.accumulate,inner_load s P p _ _ _ _ _ _ input.1,
    inner_load s G g _ _ _ _ _ _ input.2,innerModel,lane_succ]
  by_cases hr : row*C+i.1.val<rows <;> by_cases hn : c*B+i.2.1.val<N <;>
    simp only [hr,hn,and_self,and_true,and_false,true_and,false_and,if_true,if_false,
      Ideal.mul_coe,Ideal.add_coe,mul_zero,add_zero]

theorem noninner_step (s : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row col rows N K B C c : Nat) (hr : row<rows) (input : Input P G p g (rows*N*K) s) :
    FPBackwardNoninner.accumulate Ideal.algebra s P G row col rows N K B C (c*B)
      (noninnerModel p g row col N K B C c)=noninnerModel p g row col N K B C (c+1) := by
  funext i
  simp only [FPBackwardNoninner.accumulate,noninner_load s P p _ _ _ _ _ _ _ _ hr input.1,
    noninner_load s G g _ _ _ _ _ _ _ _ hr input.2,noninnerModel,lane_succ]
  by_cases hk : col*C+i.2.1.val<K <;> by_cases hn : c*B+i.1.val<N <;>
    simp only [hk,hn,and_self,and_true,and_false,true_and,false_and,if_true,if_false,
      Ideal.mul_coe,Ideal.add_coe,mul_zero,add_zero]

theorem inner_walk (s : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row rows N B C : Nat) (hB : 0<B) (input : Input P G p g (rows*N) s) :
    (FPLoop.walk (FPBackwardInner.accumulate Ideal.algebra s P G row rows N B C)
      N B 0 (fun _ => FPReference.literal Ideal.algebra 0)).2=
        innerModel p g row rows N B C (RealAddress.rounds N B) := by
  have seed : (fun _ => FPReference.literal Ideal.algebra 0)=innerModel p g row rows N B C 0 := by
    funext i; simp [innerModel]
  rw [seed]
  simpa only [Nat.zero_mul] using MathLoop.walk_exact _ _ N B 0 hB (by omega) (fun c _ => inner_step s P G p g row rows N B C c input)

theorem noninner_walk (s : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row col rows N K B C : Nat) (hB : 0<B) (hr : row<rows) (input : Input P G p g (rows*N*K) s) :
    (FPLoop.walk (FPBackwardNoninner.accumulate Ideal.algebra s P G row col rows N K B C)
      N B 0 (fun _ => FPReference.literal Ideal.algebra 0)).2=
        noninnerModel p g row col N K B C (RealAddress.rounds N B) := by
  have seed : (fun _ => FPReference.literal Ideal.algebra 0)=noninnerModel p g row col N K B C 0 := by
    funext i; simp [noninnerModel]
  rw [seed]
  simpa only [Nat.zero_mul] using MathLoop.walk_exact _ _ N B 0 hB (by omega) (fun c _ => noninner_step s P G p g row col rows N K B C c hr input)

theorem inner_reduced (p g : Nat → ℝ) (row rows N B C : Nat) (hB : 0<B) (r : Fin C)
    (hr : row*C+r.val<rows) :
    FPBackwardInner.reduced Ideal.algebra B C (innerModel p g row rows N B C (RealAddress.rounds N B)) (r,PUnit.unit)=
      ((∑ n : Fin N, p ((row*C+r.val)*N+n.val)*g ((row*C+r.val)*N+n.val) : ℝ) : WithBot ℝ) := by
  change Ideal.sum (fun i : Fin B => (lane N B (RealAddress.rounds N B) i.val
    (fun n => if row*C+r.val<rows then p ((row*C+r.val)*N+n)*g ((row*C+r.val)*N+n) else 0) : WithBot ℝ)) = _
  simp only [if_pos hr,Ideal.sum_coe,lane_total N B hB]

theorem noninner_reduced (p g : Nat → ℝ) (row col N K B C : Nat) (hB : 0<B) (k : Fin C)
    (hk : col*C+k.val<K) :
    FPBackwardNoninner.reduced Ideal.algebra B C (noninnerModel p g row col N K B C (RealAddress.rounds N B)) (k,PUnit.unit)=
      ((∑ n : Fin N, p (row*N*K+n.val*K+(col*C+k.val))*g (row*N*K+n.val*K+(col*C+k.val)) : ℝ) : WithBot ℝ) := by
  change Ideal.sum (fun i : Fin B => (lane N B (RealAddress.rounds N B) i.val
    (fun n => if col*C+k.val<K then p (row*N*K+n*K+(col*C+k.val))*g (row*N*K+n*K+(col*C+k.val)) else 0) : WithBot ℝ)) = _
  simp only [if_pos hk,Ideal.sum_coe,lane_total N B hB]

#axiomsClean inner_walk
#axiomsClean noninner_walk
#axiomsClean inner_reduced
#axiomsClean noninner_reduced
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
