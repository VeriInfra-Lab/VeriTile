import bench.tritonbench_g.attention_kernel_aligned.Proofs.RealStep
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Real
open VeriTile Triton FP.Structural Source Full RealMath
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem mass_factor (BN N base : Nat) (score v : Nat → ℝ) (m : ℝ) :
    (∑ j,mass BN N base score m j*v (base+j.val))=
      pow2 (-m)*chunk BN N base (fun j=>pow2 (score j)*v j) := by
  rw [chunk,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : base+j.val<N
  · have hpow : pow2 (score (base+j.val)-m)=pow2 (-m)*pow2 (score (base+j.val)) := by
      rw [←pow2_add]
      congr 1
      ring
    simp [mass,hj,hpow,mul_assoc]
  · simp [mass,hj]

theorem mass_sum (BN N base : Nat) (score : Nat → ℝ) (m : ℝ) :
    (∑ j,mass BN N base score m j)=pow2 (-m)*chunk BN N base (fun j=>pow2 (score j)) := by
  simpa using mass_factor BN N base score (fun _=>1) m

def Consistent (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : QLane p) (base : Nat) (a : Accumulator R p) : Prop :=
  ∃ m,a.maximum (i.1,())=some m ∧
    a.denominator (i.1,())=some (pow2 (-m)*prefixSum p.length base (fun j=>pow2 (logicalScore p s μ i.1 j))) ∧
    a.numerator i=some (pow2 (-m)*prefixSum p.length base
      (fun j=>pow2 (logicalScore p s μ i.1 j)*vAt p s μ j i.2.1.val))

theorem initial_step (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : QLane p) (hi : row p s i.1<p.length) (hBN : 0<p.blockN) :
    Consistent p s μ i p.blockN (advance model p s 0 (initialAccumulator model p)) := by
  obtain ⟨m,hm⟩:=maximum_finite p s μ h 0 i.1 (initialAccumulator model p) hi hBN (by omega)
  refine ⟨m,hm,?_,?_⟩
  · rw [normalizer_real p s μ h 0 i.1 (initialAccumulator model p) hi hBN m 0 hm (by rfl)]
    simp only [zero_mul,zero_add]
    rw [mass_sum]
    congr 2
    try simpa [prefixSum] using (prefix_step p.blockN p.length 0 (fun j=>pow2 (logicalScore p s μ i.1 j))).symm
  · rw [accumulator_real p s μ h 0 i (initialAccumulator model p) hi hBN m 0 hm (by rfl)]
    simp only [zero_mul,zero_add]
    have hf:=mass_factor p.blockN p.length 0 (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m
    simp only [Nat.zero_add] at hf
    rw [hf]
    congr 2
    try simpa [prefixSum] using (prefix_step p.blockN p.length 0 (fun j=>pow2 (logicalScore p s μ i.1 j)*vAt p s μ j i.2.1.val)).symm

theorem consistent_step (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : QLane p) (hi : row p s i.1<p.length) (hBN : 0<p.blockN)
    (c : Nat) (hr : c*p.blockN<p.length) (a : Accumulator R p) (hc : Consistent p s μ i (c*p.blockN) a) :
    Consistent p s μ i ((c+1)*p.blockN) (advance model p s c a) := by
  obtain ⟨old,ho,hl,hv⟩:=hc
  obtain ⟨m,hm⟩:=maximum_finite p s μ h c i.1 a hi hBN hr
  have key : pow2 (-old)*pow2 (old-m)=pow2 (-m) := by
    rw [←pow2_add]
    congr 1
    ring
  refine ⟨m,hm,?_,?_⟩
  · rw [normalizer_real p s μ h c i.1 a hi hBN m _ hm hl,ho]
    simp only [alpha]
    rw [mass_sum,Nat.add_mul,Nat.one_mul,prefix_step]
    congr 1
    rw [mul_right_comm (pow2 (-old)),key]
    ring
  · rw [accumulator_real p s μ h c i a hi hBN m _ hm hv,ho]
    simp only [alpha]
    rw [mass_factor p.blockN p.length (c*p.blockN) (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m,Nat.add_mul,Nat.one_mul,prefix_step]
    congr 1
    rw [mul_right_comm (pow2 (-old)),key]
    ring

theorem loop_consistent (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : QLane p) (hi : row p s i.1<p.length) (hBN : 0<p.blockN)
    (c : Nat) (hc0 : 0<c) (hc : c≤p.blocks) :
    Consistent p s μ i (c*p.blockN) (rounds model p s c) := by
  induction c with
  | zero=>omega
  | succ c ih=>
    by_cases hz : c=0
    · subst c
      simpa [rounds] using initial_step p s μ h i hi hBN
    · exact consistent_step p s μ h i hi hBN c
        ((block_guard p c hBN).mpr (by omega)) _ (ih (by omega) (by omega))

theorem answer_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : QLane p) (hi : row p s i.1<p.length) (hBN : 0<p.blockN) :
    outputValue model p s i=some
      (RealMath.expected p.length p.head p.blockN p.biasLast p.scale
        (qAt p s μ) (kAt p s μ) (vAt p s μ) (biasAt p s μ) (row p s i.1) i.2.1.val) := by
  have hc0 : 0<p.blocks := (block_guard p 0 hBN).mp (by omega)
  have hn : p.length≤p.blocks*p.blockN := by
    by_contra hn
    have:= (block_guard p p.blocks hBN).mp (by omega)
    omega
  obtain ⟨m,hm,hl,ha⟩:=loop_consistent p s μ h i hi hBN p.blocks hc0 (by rfl)
  change WithBot.realDiv ((rounds model p s p.blocks).numerator i)
    ((rounds model p s p.blocks).denominator (i.1,()))=_
  rw [ha,hl,prefix_done p.length _ _ hn,prefix_done p.length _ _ hn]
  change some ((pow2 (-m)*_) / (pow2 (-m)*_))=some _
  rw [mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]
  rfl

#axiomsClean answer_real
#specNonCircular Consistent avoiding [FP.Structural.exec,rounds]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Real
