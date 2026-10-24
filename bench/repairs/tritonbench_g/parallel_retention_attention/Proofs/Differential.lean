import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Specification

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

def lossSpec (T DK DV h : Nat) (q k : Fin T → Fin DK → ℝ)
    (v g : Fin T → Fin DV → ℝ) : ℝ :=
  ∑ t : Fin T,∑ f : Fin DV,g t f*outputSpec T DK DV h q k v t f

theorem loss_expansion (T DK DV h : Nat) (q k : Fin T → Fin DK → ℝ)
    (v g : Fin T → Fin DV → ℝ) :
    lossSpec T DK DV h q k v g=
      ∑ t : Fin T,∑ n : Fin T,∑ e : Fin DK,∑ f : Fin DV,
        (Real.sqrt (DK:ℝ))⁻¹*weight h t n*g t f*(q t e*k n e*v n f) := by
  unfold lossSpec outputSpec
  simp_rw [Finset.mul_sum,Finset.sum_mul]
  simp_rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro t ht
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro n hn
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro e he
  apply Finset.sum_congr rfl
  intro f hf
  ring

theorem affine_derivative (x dx : ℝ) : HasDerivAt (fun z : ℝ => x+z*dx) dx 0 := by
  simpa using ((hasDerivAt_id (0:ℝ)).mul_const dx).const_add x

theorem product_derivative (c q k v dq dk dv : ℝ) :
    HasDerivAt (fun z : ℝ => c*((q+z*dq)*(k+z*dk)*(v+z*dv)))
      (c*(dq*k*v+q*dk*v+q*k*dv)) 0 := by
  convert (((affine_derivative q dq).mul (affine_derivative k dk)).mul
    (affine_derivative v dv)).const_mul c using 1 <;> dsimp <;> ring

/-- A calculus theorem for the independent forward objective. -/
theorem loss_directional_derivative (T DK DV h : Nat) (q k dq dk : Fin T → Fin DK → ℝ)
    (v g dv : Fin T → Fin DV → ℝ) :
    HasDerivAt (fun z : ℝ => lossSpec T DK DV h
      (fun t e => q t e+z*dq t e) (fun n e => k n e+z*dk n e)
      (fun n f => v n f+z*dv n f) g)
      (directionalSpec T DK DV h q k dq dk v g dv) 0 := by
  simp_rw [loss_expansion]
  unfold directionalSpec
  apply HasDerivAt.fun_sum
  intro t ht
  apply HasDerivAt.fun_sum
  intro n hn
  apply HasDerivAt.fun_sum
  intro e he
  apply HasDerivAt.fun_sum
  intro f hf
  exact product_derivative _ _ _ _ _ _ _

theorem independent_vjp (T DK DV h : Nat) (q k dq dk : Fin T → Fin DK → ℝ)
    (v g dv : Fin T → Fin DV → ℝ) :
    HasDerivAt (fun z : ℝ => lossSpec T DK DV h
      (fun t e => q t e+z*dq t e) (fun n e => k n e+z*dk n e)
      (fun n f => v n f+z*dv n f) g)
      ((∑ t : Fin T,∑ e : Fin DK,dq t e*querySpec T DK DV h k v g t e)+
       (∑ n : Fin T,∑ e : Fin DK,dk n e*keySpec T DK DV h q v g n e)+
       (∑ n : Fin T,∑ f : Fin DV,dv n f*valueSpec T DK DV h q k g n f)) 0 := by
  rw [← vjp_pairing]
  exact loss_directional_derivative T DK DV h q k dq dk v g dv

#axiomsClean loss_expansion
#axiomsClean loss_directional_derivative
#axiomsClean independent_vjp
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
