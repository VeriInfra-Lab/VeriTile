import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardModel

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

theorem inner_flat (p g : Nat → ℝ) (row N n : Nat) (hn : n<N) :
    Target.flatBackward p g N 1 (row*N+n)=
      p (row*N+n)*(g (row*N+n)-∑ j : Fin N, p (row*N+j.val)*g (row*N+j.val)) := by
  have h := Target.coordinates N 1 row 0 n (by omega) (by decide) hn (by decide)
  simp only [Target.address,Nat.mul_one,Nat.add_zero] at h
  simp only [Target.flatBackward,h.1,h.2,Target.address,Nat.mul_one,Nat.add_zero]

theorem noninner_flat (p g : Nat → ℝ) (row N K k n : Nat) (hn : n<N) (hk : k<K) :
    Target.flatBackward p g N K (row*N*K+n*K+k)=
      p (row*N*K+n*K+k)*(g (row*N*K+n*K+k)-∑ j : Fin N, p (row*N*K+j.val*K+k)*g (row*N*K+j.val*K+k)) := by
  have h := Target.coordinates N K row k n (by omega) (by omega) hn hk
  change (row*N*K+n*K+k)/K/N=row ∧ (row*N*K+n*K+k)%K=k at h
  simp only [Target.flatBackward,h.1,h.2,Target.address]

theorem online_inner_value (t : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row rows N B C cur : Nat) (hB : 0<B) (input : Input P G p g (rows*N) t)
    (i : TileIndex [C,B]) (hr : row*C+i.1.val<rows) (hn : cur+i.2.1.val<N) :
    FPBackwardInner.storedValue Ideal.algebra t P G row rows N B C cur
      (FPBackwardInner.reduced Ideal.algebra B C (innerModel p g row rows N B C (RealAddress.rounds N B))) i=
      (Target.flatBackward p g N 1 (FPBackwardInner.offset row N cur i) : WithBot ℝ) := by
  rw [FPBackwardInner.storedValue,inner_load t P p _ _ _ _ _ _ input.1,
    inner_load t G g _ _ _ _ _ _ input.2,inner_reduced p g row rows N B C hB i.1 hr]
  simp only [hr,hn,and_self,if_true,Ideal.sub_coe,Ideal.mul_coe,Ideal.cast_value,
    FPBackwardInner.offset,inner_flat p g _ _ _ hn]

theorem online_noninner_value (t : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row col rows N K B C cur : Nat) (hB : 0<B) (hr : row<rows) (input : Input P G p g (rows*N*K) t)
    (i : TileIndex [B,C]) (hn : cur+i.1.val<N) (hk : col*C+i.2.1.val<K) :
    FPBackwardNoninner.storedValue Ideal.algebra t P G row col rows N K B C cur
      (FPBackwardNoninner.reduced Ideal.algebra B C (noninnerModel p g row col N K B C (RealAddress.rounds N B))) i=
      (Target.flatBackward p g N K (FPBackwardNoninner.offset row col N K cur i) : WithBot ℝ) := by
  rw [FPBackwardNoninner.storedValue,noninner_load t P p _ _ _ _ _ _ _ _ hr input.1,
    noninner_load t G g _ _ _ _ _ _ _ _ hr input.2,noninner_reduced p g row col N K B C hB i.2.1 hk]
  simp only [hn,hk,and_self,if_true,Ideal.sub_coe,Ideal.mul_coe,Ideal.cast_value,
    FPBackwardNoninner.offset,noninner_flat p g _ _ _ _ _ hn hk]

theorem one_inner_load (t : State (WithBot ℝ)) (R : RegionName) (v : Nat → ℝ)
    (row rows N B C : Nat) (read : Read R v (rows*N) t) (i : TileIndex [C,B]) :
    FPReference.backwardInnerLoad Ideal.algebra t R row rows N B C i=
      ((if row*C+i.1.val<rows ∧ i.2.1.val<N then v ((row*C+i.1.val)*N+i.2.1.val) else 0 : ℝ) : WithBot ℝ) := by
  simpa only [FPBackwardInner.load,FPBackwardInner.active,FPBackwardInner.offset,
    FPReference.backwardInnerLoad,FPReference.backwardInnerActive,FPReference.backwardInnerOffset,Nat.zero_add]
    using inner_load t R v row rows N B C 0 read i

theorem one_noninner_load (t : State (WithBot ℝ)) (R : RegionName) (v : Nat → ℝ)
    (row col rows N K B C : Nat) (hr : row<rows) (read : Read R v (rows*N*K) t) (i : TileIndex [B,C]) :
    FPReference.backwardNoninnerLoad Ideal.algebra t R row col N K B C i=
      ((if i.1.val<N ∧ col*C+i.2.1.val<K then v (row*N*K+i.1.val*K+(col*C+i.2.1.val)) else 0 : ℝ) : WithBot ℝ) := by
  simpa only [FPBackwardNoninner.load,FPBackwardNoninner.active,FPBackwardNoninner.offset,
    FPReference.backwardNoninnerLoad,FPReference.noninnerActive,FPReference.noninnerOffset,Nat.zero_add]
    using noninner_load t R v row col rows N K B C 0 hr read i

theorem one_inner_scale (t : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row rows N B C : Nat) (hB : 0<B) (covered : N≤B) (input : Input P G p g (rows*N) t)
    (r : Fin C) (hr : row*C+r.val<rows) :
    FPReference.backwardInnerScale Ideal.algebra t P G row rows N B C (r,PUnit.unit)=
      ((∑ n : Fin N, p ((row*C+r.val)*N+n.val)*g ((row*C+r.val)*N+n.val) : ℝ) : WithBot ℝ) := by
  change Ideal.sum (fun i : Fin B => FPReference.mul Ideal.algebra
    (FPReference.backwardInnerLoad Ideal.algebra t P row rows N B C (r,i,PUnit.unit))
    (FPReference.backwardInnerLoad Ideal.algebra t G row rows N B C (r,i,PUnit.unit))) = _
  have terms (i : Fin B) : FPReference.mul Ideal.algebra
      (FPReference.backwardInnerLoad Ideal.algebra t P row rows N B C (r,i,PUnit.unit))
      (FPReference.backwardInnerLoad Ideal.algebra t G row rows N B C (r,i,PUnit.unit))=
      ((if i.val<N then p ((row*C+r.val)*N+i.val)*g ((row*C+r.val)*N+i.val) else 0 : ℝ) : WithBot ℝ) := by
    rw [one_inner_load t P p _ _ _ _ _ input.1,one_inner_load t G g _ _ _ _ _ input.2]
    by_cases hn : i.val<N <;> simp only [hr,hn,and_self,and_false,if_true,if_false,Ideal.mul_coe,mul_zero]
  simp_rw [terms]
  rw [Ideal.sum_coe]
  exact congrArg (fun x : ℝ => (x : WithBot ℝ)) (RealForward.sum_padding N B hB covered
    (fun n => p ((row*C+r.val)*N+n)*g ((row*C+r.val)*N+n)))

theorem one_noninner_scale (t : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row col rows N K B C : Nat) (hB : 0<B) (covered : N≤B) (hr : row<rows)
    (input : Input P G p g (rows*N*K) t) (k : Fin C) (hk : col*C+k.val<K) :
    FPReference.backwardNoninnerScale Ideal.algebra t P G row col N K B C (k,PUnit.unit)=
      ((∑ n : Fin N, p (row*N*K+n.val*K+(col*C+k.val))*g (row*N*K+n.val*K+(col*C+k.val)) : ℝ) : WithBot ℝ) := by
  change Ideal.sum (fun i : Fin B => FPReference.mul Ideal.algebra
    (FPReference.backwardNoninnerLoad Ideal.algebra t P row col N K B C (i,k,PUnit.unit))
    (FPReference.backwardNoninnerLoad Ideal.algebra t G row col N K B C (i,k,PUnit.unit))) = _
  have terms (i : Fin B) : FPReference.mul Ideal.algebra
      (FPReference.backwardNoninnerLoad Ideal.algebra t P row col N K B C (i,k,PUnit.unit))
      (FPReference.backwardNoninnerLoad Ideal.algebra t G row col N K B C (i,k,PUnit.unit))=
      ((if i.val<N then p (row*N*K+i.val*K+(col*C+k.val))*g (row*N*K+i.val*K+(col*C+k.val)) else 0 : ℝ) : WithBot ℝ) := by
    rw [one_noninner_load t P p row col rows N K B C hr input.1,one_noninner_load t G g row col rows N K B C hr input.2]
    by_cases hn : i.val<N <;> simp only [hk,hn,and_self,false_and,if_true,if_false,Ideal.mul_coe,mul_zero]
  simp_rw [terms]
  rw [Ideal.sum_coe]
  exact congrArg (fun x : ℝ => (x : WithBot ℝ)) (RealForward.sum_padding N B hB covered
    (fun n => p (row*N*K+n*K+(col*C+k.val))*g (row*N*K+n*K+(col*C+k.val))))

theorem one_inner_value (t : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row rows N B C : Nat) (hB : 0<B) (covered : N≤B) (input : Input P G p g (rows*N) t)
    (i : TileIndex [C,B]) (hr : row*C+i.1.val<rows) (hn : i.2.1.val<N) :
    FPReference.backwardInnerValue Ideal.algebra t P G row rows N B C i=
      (Target.flatBackward p g N 1 (FPReference.backwardInnerOffset row N i) : WithBot ℝ) := by
  rw [FPReference.backwardInnerValue,one_inner_load t P p _ _ _ _ _ input.1,
    one_inner_load t G g _ _ _ _ _ input.2,one_inner_scale t P G p g row rows N B C hB covered input i.1 hr]
  simp only [hr,hn,and_self,if_true,Ideal.sub_coe,Ideal.mul_coe,Ideal.cast_value,
    FPReference.backwardInnerOffset,inner_flat p g _ _ _ hn]

theorem one_noninner_value (t : State (WithBot ℝ)) (P G : RegionName) (p g : Nat → ℝ)
    (row col rows N K B C : Nat) (hB : 0<B) (covered : N≤B) (hr : row<rows)
    (input : Input P G p g (rows*N*K) t) (i : TileIndex [B,C]) (hn : i.1.val<N) (hk : col*C+i.2.1.val<K) :
    FPReference.backwardNoninnerValue Ideal.algebra t P G row col N K B C i=
      (Target.flatBackward p g N K (FPReference.noninnerOffset row col N K i) : WithBot ℝ) := by
  rcases i with ⟨n,k,u⟩; cases u
  rw [FPReference.backwardNoninnerValue,one_noninner_load t P p row col rows N K B C hr input.1,
    one_noninner_load t G g row col rows N K B C hr input.2,
    one_noninner_scale t P G p g row col rows N K B C hB covered hr input k hk]
  simp only [hn,hk,and_self,if_true,Ideal.sub_coe,Ideal.mul_coe,Ideal.cast_value,
    FPReference.noninnerOffset,noninner_flat p g _ _ _ _ _ hn hk]

#axiomsClean online_inner_value
#axiomsClean online_noninner_value
#axiomsClean one_inner_value
#axiomsClean one_noninner_value
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
