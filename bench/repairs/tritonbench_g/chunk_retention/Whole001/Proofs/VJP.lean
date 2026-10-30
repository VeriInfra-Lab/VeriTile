import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Math

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.MathTarget
noncomputable section
open scoped BigOperators

/-- Extend a finite prefix by explicitly zeroed inactive terms. -/
theorem sum_prefix (f : Nat → ℝ) (small large : Nat) (h : small≤large) :
    (∑ i ∈ Finset.range small, f i)=
      ∑ i ∈ Finset.range large, if i<small then f i else 0 := by
  calc
    _ = ∑ i ∈ Finset.range small, if i<small then f i else 0 := by
      apply Finset.sum_congr rfl
      intro i hi
      simp [Finset.mem_range.mp hi]
    _ = _ := Finset.sum_subset (Finset.range_mono h) (by
      intro i hi hnot
      simp only [Finset.mem_range] at hnot
      simp [hnot])

theorem sum_shift (f : Nat → ℝ) (offset total : Nat) (h : offset≤total) :
    (∑ i ∈ Finset.range (total-offset), f (offset+i))=
      ∑ i ∈ Finset.range total, if (offset ≤ i) then f i else 0 := by
  have he : total=offset+(total-offset) := by omega
  conv_rhs => rw [he,Finset.sum_range_add]
  have hzero : (∑ i ∈ Finset.range offset, if (offset ≤ i) then f i else 0)=0 := by
    apply Finset.sum_eq_zero
    intro i hi
    simp [Nat.not_le.mpr (Finset.mem_range.mp hi)]
  rw [hzero,zero_add]
  apply Finset.sum_congr rfl
  intro i hi
  simp

def denseCoefficient (rate terminal : ℝ) (forcing : Nat → ℝ) (time total : Nat) : ℝ :=
  rate^(total-1-time)*terminal+
    ∑ row ∈ Finset.range total, if time≤row then rate^(row-time)*forcing row else 0

theorem coefficient_dense (rate terminal : ℝ) (forcing : Nat → ℝ)
    (time total : Nat) (ht : time<total) :
    coefficient rate terminal forcing time (total-time)=
      denseCoefficient rate terminal forcing time total := by
  unfold coefficient denseCoefficient
  rw [show total-time-1=total-1-time by omega]
  congr 1
  have hs := sum_shift (fun row => rate^(row-time)*forcing row) time total (Nat.le_of_lt ht)
  simpa using hs

/-- Finite reverse-mode identity for the independent input-only prefix.
This is the full time dimension, including the terminal loss and seed variation. -/
theorem prefix_adjoint (rate terminal dSeed : ℝ) (forcing delta : Nat → ℝ) (total : Nat) :
    (∑ row ∈ Finset.range total, forcing row*weighted rate dSeed delta (row+1))+
      terminal*weighted rate dSeed delta total=
    suffix rate terminal forcing 0 total*dSeed+
      ∑ time ∈ Finset.range total, denseCoefficient rate terminal forcing time total*delta time := by
  have hinner (row : Nat) (hr : row<total) :
      (∑ time ∈ Finset.range (row+1), rate^(row+1-1-time)*delta time)=
      ∑ time ∈ Finset.range total, if time≤row then rate^(row-time)*delta time else 0 := by
    have h := sum_prefix (fun time => rate^(row-time)*delta time) (row+1) total (by omega)
    simpa only [Nat.add_sub_cancel,Nat.lt_succ_iff] using h
  have hdouble :
      (∑ row ∈ Finset.range total, forcing row*
        ∑ time ∈ Finset.range (row+1), rate^(row+1-1-time)*delta time)=
      ∑ time ∈ Finset.range total,
        (∑ row ∈ Finset.range total, if time≤row then rate^(row-time)*forcing row else 0)*delta time := by
    calc
      _ = ∑ row ∈ Finset.range total, forcing row*
          ∑ time ∈ Finset.range total, if time≤row then rate^(row-time)*delta time else 0 := by
            apply Finset.sum_congr rfl
            intro row hr
            rw [hinner row (Finset.mem_range.mp hr)]
      _ = ∑ time ∈ Finset.range total, ∑ row ∈ Finset.range total,
          forcing row*(if time≤row then rate^(row-time)*delta time else 0) := by
            simp_rw [Finset.mul_sum]
            rw [Finset.sum_comm]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro time ht
        rw [Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro row hr
        split_ifs <;> ring
  unfold weighted
  simp_rw [mul_add,Finset.sum_add_distrib]
  rw [hdouble]
  unfold suffix denseCoefficient
  simp only [mul_add,add_mul,Finset.sum_add_distrib,Finset.sum_mul,Finset.mul_sum,Nat.zero_add]
  have hseed : (∑ row ∈ Finset.range total, forcing row*(rate^(row+1)*dSeed))=
      ∑ row ∈ Finset.range total, rate^(row+1)*forcing row*dSeed := by
    apply Finset.sum_congr rfl
    intro i hi
    ring
  have hterminal : (∑ time ∈ Finset.range total, terminal*(rate^(total-1-time)*delta time))=
      ∑ time ∈ Finset.range total, rate^(total-1-time)*terminal*delta time := by
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hseed,hterminal]
  ring

/-- The derivative of one scalar state coordinate, with all time terms. -/
def stateVariation (rate seed : ℝ) (k v dk dv : Nat → ℝ) (time : Nat) :=
  weighted rate seed (fun t => dk t*v t+k t*dv t) time

/-- The coefficient of ε in the full output/terminal loss for one state coordinate.
This definition differentiates the polynomial input expression directly. -/
def lossVariation (rate seed dSeed terminal scale : ℝ)
    (q k v incoming dq dk dv : Nat → ℝ) (total : Nat) :=
  (∑ t ∈ Finset.range total, incoming t*scale*
    (dq t*weighted rate seed (fun u => k u*v u) (t+1)+
      q t*stateVariation rate dSeed k v dk dv (t+1)))+
    terminal*stateVariation rate dSeed k v dk dv total

theorem coordinate_vjp (rate seed dSeed terminal scale : ℝ)
    (q k v incoming dq dk dv : Nat → ℝ) (total : Nat) :
    lossVariation rate seed dSeed terminal scale q k v incoming dq dk dv total=
    (∑ t ∈ Finset.range total,
      (scale*incoming t*weighted rate seed (fun u => k u*v u) (t+1))*dq t)+
    (∑ t ∈ Finset.range total,
      (denseCoefficient rate terminal (fun u => scale*q u*incoming u) t total*v t)*dk t)+
    (∑ t ∈ Finset.range total,
      (denseCoefficient rate terminal (fun u => scale*q u*incoming u) t total*k t)*dv t)+
    suffix rate terminal (fun u => scale*q u*incoming u) 0 total*dSeed := by
  unfold lossVariation stateVariation
  have hsplit : (∑ t ∈ Finset.range total, incoming t*scale*
      (dq t*weighted rate seed (fun u => k u*v u) (t+1)+
      q t*weighted rate dSeed (fun u => dk u*v u+k u*dv u) (t+1)))=
    (∑ t ∈ Finset.range total,
      (scale*incoming t*weighted rate seed (fun u => k u*v u) (t+1))*dq t)+
    (∑ t ∈ Finset.range total,
      (scale*q t*incoming t)*weighted rate dSeed (fun u => dk u*v u+k u*dv u) (t+1)) := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hsplit,add_assoc,prefix_adjoint]
  have hsplit2 : (∑ t ∈ Finset.range total,
      denseCoefficient rate terminal (fun u => scale*q u*incoming u) t total*(dk t*v t+k t*dv t))=
    (∑ t ∈ Finset.range total,
      (denseCoefficient rate terminal (fun u => scale*q u*incoming u) t total*v t)*dk t)+
    (∑ t ∈ Finset.range total,
      (denseCoefficient rate terminal (fun u => scale*q u*incoming u) t total*k t)*dv t) := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hsplit2]
  ring

#specNonCircular lossVariation avoiding [prefix_adjoint,coordinate_vjp]
#axiomsClean prefix_adjoint
#axiomsClean coordinate_vjp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.MathTarget
