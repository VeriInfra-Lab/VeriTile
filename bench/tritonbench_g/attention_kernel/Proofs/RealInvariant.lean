import bench.tritonbench_g.attention_kernel.Proofs.RealStep
import bench.tritonbench_g.attention_kernel.Proofs.Grid
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Real
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def prefixSum (N t : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j∈Finset.range t, if j<N then f j else 0

def chunk (BN N row : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j : Fin BN, if row+j.val<N then f (row+j.val) else 0

theorem prefix_step (BN N row : Nat) (f : Nat → ℝ) :
    prefixSum N (row+BN) f=prefixSum N row f+chunk BN N row f := by
  simp only [prefixSum,chunk,Finset.sum_range_add]
  rw [Fin.sum_univ_eq_sum_range (fun j=>if row+j<N then f (row+j) else 0) BN]

theorem prefix_done (N row : Nat) (f : Nat → ℝ) (h : N≤row) :
    prefixSum N row f=∑ j∈Finset.range N,f j := by
  rw [show row=N+(row-N) by omega,prefixSum,Finset.sum_range_add]
  have first : (∑ j∈Finset.range N, if j<N then f j else 0)=∑ j∈Finset.range N,f j := by
    apply Finset.sum_congr rfl
    intro j hj
    simp [Finset.mem_range.mp hj]
  simpa [first]

theorem mass_factor (BN N row : Nat) (score v : Nat → ℝ) (m : ℝ) :
    (∑ j,mass BN N row score m j*v (row+j.val))=
      pow2 (-m)*chunk BN N row (fun j=>pow2 (score j)*v j) := by
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

theorem mass_sum (BN N row : Nat) (score : Nat → ℝ) (m : ℝ) :
    (∑ j,mass BN N row score m j)=pow2 (-m)*chunk BN N row (fun j=>pow2 (score j)) := by
  simpa using mass_factor BN N row score (fun _=>1) m

def Consistent (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : TileIndex [p.bm,p.dim]) (row : Nat) (a : Acc p R) : Prop :=
  ∃ m,a.maximum i.1=some m ∧
    a.normalizer i.1=some (pow2 (-m)*prefixSum p.length row (fun j=>pow2 (logicalScore p s μ i.1 j))) ∧
    a.value i=some (pow2 (-m)*prefixSum p.length row
      (fun j=>pow2 (logicalScore p s μ i.1 j)*vAt p s μ j i.2.1.val))

theorem initial_step (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [p.bm,p.dim]) (hi : s.pids 0*p.bm+i.1.val<p.length) (hbn : 0<p.bn) :
    Consistent p s μ i p.bn (update algebra dotModel p s 0 (initialAcc algebra p)) := by
  obtain ⟨m,hm⟩:=maximum_finite p s μ h 0 i.1 (initialAcc algebra p) hi hbn (by simp) (by omega)
  refine ⟨m,hm,?_,?_⟩
  · rw [normalizer_real p s μ h 0 i.1 (initialAcc algebra p) hi hbn (by simp) m 0 hm (by norm_num [initialAcc,algebra,zero])]
    have ha : (initialAcc algebra p).maximum i.1=none := rfl
    rw [ha]
    simp only [alpha,mul_zero,zero_add]
    rw [mass_sum p.bn]
    congr 2
    try simpa [prefixSum] using (prefix_step p.bn p.length 0 (fun j=>pow2 (logicalScore p s μ i.1 j))).symm
  · rw [accumulator_real p s μ h 0 i (initialAcc algebra p) hi hbn (by simp) m 0 hm (by rfl)]
    simp only [initialAcc,zero_mul,zero_add]
    have hf:=mass_factor p.bn p.length 0 (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m
    simp only [Nat.zero_add] at hf
    rw [hf]
    congr 2
    try simpa [prefixSum] using (prefix_step p.bn p.length 0 (fun j=>pow2 (logicalScore p s μ i.1 j)*vAt p s μ j i.2.1.val)).symm

theorem consistent_step (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [p.bm,p.dim]) (hi : s.pids 0*p.bm+i.1.val<p.length) (hbn : 0<p.bn)
    (row : Nat) (ha : row%p.bn=0) (hr : row<p.length) (a : Acc p R) (hc : Consistent p s μ i row a) :
    Consistent p s μ i (row+p.bn) (update algebra dotModel p s row a) := by
  obtain ⟨old,ho,hl,hv⟩:=hc
  obtain ⟨m,hm⟩:=maximum_finite p s μ h row i.1 a hi hbn ha hr
  have key : pow2 (-old)*pow2 (old-m)=pow2 (-m) := by
    rw [←pow2_add]
    congr 1
    ring
  refine ⟨m,hm,?_,?_⟩
  · rw [normalizer_real p s μ h row i.1 a hi hbn ha m _ hm hl,ho]
    simp only [alpha]
    rw [mass_sum p.bn,prefix_step p.bn]
    congr 1
    rw [mul_right_comm (pow2 (-old)),key]
    ring
  · rw [accumulator_real p s μ h row i a hi hbn ha m _ hm hv,ho]
    simp only [alpha]
    rw [mass_factor p.bn p.length row (logicalScore p s μ i.1) (fun j=>vAt p s μ j i.2.1.val) m,prefix_step p.bn]
    congr 1
    rw [mul_right_comm (pow2 (-old)),key]
    ring

theorem loop_consistent (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [p.bm,p.dim]) (hi : s.pids 0*p.bm+i.1.val<p.length) (hbn : 0<p.bn)
    (row : Nat) (ha : row%p.bn=0) (a : Acc p R) (hc : Consistent p s μ i row a) :
    ∃ last,p.length≤last ∧ Consistent p s μ i last
      (foldRange (update algebra dotModel p s) row p.length p.bn a) := by
  by_cases hr : row<p.length
  · obtain ⟨last,hl,hc⟩:=loop_consistent p s μ h i hi hbn (row+p.bn) (by simp [Nat.add_mod,ha])
      (update algebra dotModel p s row a) (consistent_step p s μ h i hi hbn row ha hr a hc)
    refine ⟨last,hl,?_⟩
    rw [foldRange]
    simpa only [show ¬p.bn=0 by omega,hr,if_false,if_true] using hc
  · refine ⟨row,by omega,?_⟩
    rw [foldRange]
    simpa only [show ¬p.bn=0 by omega,hr,if_false] using hc
termination_by p.length-row
decreasing_by omega

theorem answer_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [p.bm,p.dim]) (hbn : 0<p.bn) (hi : active p s i) :
    Whole.answer algebra dotModel p s i=some
      (RealMath.expected p.length p.dim p.bn p.scale (qAt p s μ) (kAt p s μ) (vAt p s μ)
        (bAt p s μ) (s.pids 0*p.bm+i.1.val) i.2.1.val) := by
  have hn : 0<p.length := by have : s.pids 0*p.bm+i.1.val<p.length := hi;omega
  obtain ⟨last,hl,m,hm,hnorm,hacc⟩:=loop_consistent p s μ h i hi hbn p.bn (by simp)
    (update algebra dotModel p s 0 (initialAcc algebra p)) (initial_step p s μ h i hi hbn)
  have he : accumulated algebra dotModel p s=
      foldRange (update algebra dotModel p s) p.bn p.length p.bn (update algebra dotModel p s 0 (initialAcc algebra p)) := by
    rw [accumulated,foldRange]
    simp [hn,Nat.ne_of_gt hbn]
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
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Real
