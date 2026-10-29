import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RealStep
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Real
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def prefixSum (N t : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j∈Finset.range t, if j<N then f j else 0

def chunk (N row : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j : Fin 64, if row+j.val<N then f (row+j.val) else 0

theorem prefix_step (N row : Nat) (f : Nat → ℝ) :
    prefixSum N (row+64) f=prefixSum N row f+chunk N row f := by
  simp only [prefixSum,chunk,Finset.sum_range_add]
  rw [Fin.sum_univ_eq_sum_range (fun j=>if row+j<N then f (row+j) else 0) 64]

theorem prefix_done (N row : Nat) (f : Nat → ℝ) (h : N≤row) :
    prefixSum N row f=∑ j∈Finset.range N,f j := by
  rw [show row=N+(row-N) by omega,prefixSum,Finset.sum_range_add]
  have first : (∑ j∈Finset.range N, if j<N then f j else 0)=∑ j∈Finset.range N,f j := by
    apply Finset.sum_congr rfl
    intro j hj
    simp [Finset.mem_range.mp hj]
  simpa [first]

theorem mass_factor (N row : Nat) (score v : Nat → ℝ) (m : ℝ) :
    (∑ j,mass N row score m j*v (row+j.val))=
      pow2 (-m)*chunk N row (fun j=>pow2 (score j)*v j) := by
  rw [chunk,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : row+j.val<N
  · have h' : j.val<N-row := by omega
    have hpow : pow2 (score (row+j.val)-m)=pow2 (-m)*pow2 (score (row+j.val)) := by
      rw [←pow2_add]
      congr 1
      ring
    simp [mass,hj,h',hpow,mul_assoc]
  · have h' : ¬j.val<N-row := by omega
    simp [mass,hj,h']

theorem mass_sum (N row : Nat) (score : Nat → ℝ) (m : ℝ) :
    (∑ j,mass N row score m j)=pow2 (-m)*chunk N row (fun j=>pow2 (score j)) := by
  simpa using mass_factor N row score (fun _=>1) m

def Consistent (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : TileIndex [128,128]) (row : Nat) (a : Acc R) : Prop :=
  ∃ m,a.maximum i.1=some m ∧
    a.normalizer i.1=some (pow2 (-m)*prefixSum p.length row (fun j=>pow2 (logicalScore p s μ i.1 j))) ∧
    a.value i=some (pow2 (-m)*prefixSum p.length row
      (fun j=>pow2 (logicalScore p s μ i.1 j)*vAt p s μ j i.2.1.val))

theorem initial_step (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [128,128]) (hi : s.pids 0*128+i.1.val<p.length) (hd : i.2.1.val<96) :
    Consistent p s μ i 64 (update algebra dotModel p s 0 (initialAcc algebra)) := by
  obtain ⟨m,hm⟩:=maximum_finite p s μ h 0 i.1 (initialAcc algebra) hi (by omega) (by omega)
  refine ⟨m,hm,?_,?_⟩
  · rw [normalizer_real p s μ h 0 i.1 (initialAcc algebra) hi (by omega) m 1 hm (by norm_num [initialAcc,algebra,zero])]
    have ha : (initialAcc algebra).maximum i.1=none := rfl
    rw [ha]
    simp only [alpha,mul_zero,zero_add]
    rw [mass_sum]
    congr 2
    try simpa [prefixSum] using (prefix_step p.length 0 (fun j=>pow2 (logicalScore p s μ i.1 j))).symm
  · rw [accumulator_real p s μ h 0 i (initialAcc algebra) hi hd (by omega) m 0 hm (by rfl)]
    simp only [initialAcc,zero_mul,zero_add]
    have hf:=mass_factor p.length 0 (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m
    simp only [Nat.zero_add] at hf
    rw [hf]
    congr 2
    try simpa [prefixSum] using (prefix_step p.length 0 (fun j=>pow2 (logicalScore p s μ i.1 j)*vAt p s μ j i.2.1.val)).symm

theorem consistent_step (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [128,128]) (hi : s.pids 0*128+i.1.val<p.length) (hd : i.2.1.val<96)
    (row : Nat) (ha : row%64=0) (hr : row<p.length) (a : Acc R) (hc : Consistent p s μ i row a) :
    Consistent p s μ i (row+64) (update algebra dotModel p s row a) := by
  obtain ⟨old,ho,hl,hv⟩:=hc
  obtain ⟨m,hm⟩:=maximum_finite p s μ h row i.1 a hi ha hr
  have key : pow2 (-old)*pow2 (old-m)=pow2 (-m) := by
    rw [←pow2_add]
    congr 1
    ring
  refine ⟨m,hm,?_,?_⟩
  · rw [normalizer_real p s μ h row i.1 a hi ha m _ hm hl,ho]
    simp only [alpha]
    rw [mass_sum,prefix_step]
    congr 1
    rw [mul_right_comm (pow2 (-old)),key]
    ring
  · rw [accumulator_real p s μ h row i a hi hd ha m _ hm hv,ho]
    simp only [alpha]
    rw [mass_factor p.length row (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m,prefix_step]
    congr 1
    rw [mul_right_comm (pow2 (-old)),key]
    ring

theorem loop_consistent (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [128,128]) (hi : s.pids 0*128+i.1.val<p.length) (hd : i.2.1.val<96)
    (row : Nat) (ha : row%64=0) (a : Acc R) (hc : Consistent p s μ i row a) :
    ∃ last,p.length≤last ∧ Consistent p s μ i last
      (foldRange (update algebra dotModel p s) row p.length 64 a) := by
  by_cases hr : row<p.length
  · obtain ⟨last,hl,hc⟩:=loop_consistent p s μ h i hi hd (row+64) (by omega)
      (update algebra dotModel p s row a) (consistent_step p s μ h i hi hd row ha hr a hc)
    refine ⟨last,hl,?_⟩
    rw [foldRange]
    simpa only [show ¬(64:Nat)=0 by omega,hr,if_false,if_true] using hc
  · refine ⟨row,by omega,?_⟩
    rw [foldRange]
    simpa only [show ¬(64:Nat)=0 by omega,hr,if_false] using hc
termination_by p.length-row
decreasing_by omega

theorem answer_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [128,128]) (hi : active p s i) :
    Whole.answer algebra dotModel p s i=some
      (RealMath.expected p.length (qAt p s μ) (kAt p s μ) (vAt p s μ)
        (qsAt p s μ) (ksAt p s μ) (s.pids 0*128+i.1.val) i.2.1.val) := by
  have hn : 0<p.length := by have:=hi.1;omega
  obtain ⟨last,hl,m,hm,hnorm,hacc⟩:=loop_consistent p s μ h i hi.1 hi.2 64 (by omega)
    (update algebra dotModel p s 0 (initialAcc algebra)) (initial_step p s μ h i hi.1 hi.2)
  have he : accumulated algebra dotModel p s=
      foldRange (update algebra dotModel p s) 64 p.length 64 (update algebra dotModel p s 0 (initialAcc algebra)) := by
    rw [accumulated,foldRange]
    simp [hn]
  change WithBot.realDiv ((accumulated algebra dotModel p s).value i)
    ((accumulated algebra dotModel p s).normalizer i.1)=_
  rw [he,hacc,hnorm,prefix_done p.length last _ hl,prefix_done p.length last _ hl]
  change some ((pow2 (-m)*_) / (pow2 (-m)*_))=some _
  rw [mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]
  congr 1
  try simp [RealMath.expected,logicalScore,List.sum_map_range]

#axiomsClean answer_real
#specNonCircular Consistent avoiding [FP.Structural.exec,accumulated]
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Real
