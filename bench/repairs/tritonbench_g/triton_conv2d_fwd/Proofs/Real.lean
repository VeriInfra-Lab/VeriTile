import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Arithmetic
namespace VeriTile.Bench.Repairs.Conv2d
open scoped BigOperators
set_option maxHeartbeats 1600000
set_option maxRecDepth 4096

theorem dot32_real (flags : Flags) (a : Mat 64 32 ℝ) (b : Mat 32 32 ℝ)
    (carry : Frag ℝ) (i : Fin 64) (j : Fin 32) :
    dot32 realISA flags a b carry i j=(∑ k : Fin 32,a i k*b k j)+carry i j := by
  rcases flags with ⟨half,tf⟩
  cases half <;> cases tf <;>
    norm_num [dot32,mma16Step,mma8Step,fmaLanes,realISA,Fin.sum_univ_succ,Fin.succ] <;> ring

noncomputable def term (s : Shape) (m : Memory ℝ) (g r f h w c : Nat) : ℝ :=
  refA s realISA m g r h w c*refB s realISA m g f h w c

theorem sourceStep_real (s : Shape) (flags : Flags) (m : Memory ℝ)
    (g r f h w cb : Nat) (hg : g<s.groups) (carry : Frag ℝ) (i : Fin 64) (j : Fin 32) :
    sourceStep s realISA flags m g r f h w cb carry i j=
      (∑ k ∈ Finset.range 32,term s m g (r+i.val) (f+j.val) h w (cb*32+k))+carry i j := by
  unfold sourceStep
  rw [dot32_real]
  congr 1
  simp only [aLoad_reference s realISA m g _ _ _ _ hg,bLoad_reference,term]
  exact Fin.sum_univ_eq_sum_range (fun k => term s m g (r+i.val) (f+j.val) h w (cb*32+k)) 32

theorem channel_real (s : Shape) (flags : Flags) (m : Memory ℝ)
    (g r f h w cb : Nat) (hg : g<s.groups) (carry : Frag ℝ) (i : Fin 64) (j : Fin 32) :
    channelLoop s realISA flags m g r f h w cb carry i j=
      (∑ c ∈ Finset.range (cb*32),term s m g (r+i.val) (f+j.val) h w c)+carry i j := by
  induction cb with
  | zero => simp [channelLoop]
  | succ cb ih =>
    rw [channelLoop,sourceStep_real s flags m g r f h w cb hg,ih]
    rw [Nat.succ_mul,Finset.sum_range_add]
    ring

theorem width_real (s : Shape) (flags : Flags) (m : Memory ℝ)
    (g r f h w : Nat) (hg : g<s.groups) (carry : Frag ℝ) (i : Fin 64) (j : Fin 32) :
    widthLoop s realISA flags m g r f h w carry i j=
      (∑ w ∈ Finset.range w,∑ c ∈ Finset.range s.padded,
        term s m g (r+i.val) (f+j.val) h w c)+carry i j := by
  induction w with
  | zero => simp [widthLoop]
  | succ w ih =>
    rw [widthLoop,channel_real s flags m g r f h w _ hg,ih,Finset.sum_range_succ]
    simp only [Shape.padded]
    ring

theorem source_real_sum (s : Shape) (flags : Flags) (m : Memory ℝ) (x : Cell)
    (hg : x.group<s.groups) : sourceValue s realISA flags m x=
      ∑ h ∈ Finset.range s.kh,∑ w ∈ Finset.range s.kw,∑ c ∈ Finset.range s.padded,
        term s m x.group x.spatial x.feature h w c := by
  have height : ∀ n i j,heightLoop s realISA flags m x.group (x.spatial/64*64) (x.feature/32*32) n i j=
      ∑ h ∈ Finset.range n,∑ w ∈ Finset.range s.kw,∑ c ∈ Finset.range s.padded,
        term s m x.group (x.spatial/64*64+i.val) (x.feature/32*32+j.val) h w c := by
    intro n i j
    induction n with
    | zero => simp [heightLoop,realISA]
    | succ n ih =>
      rw [heightLoop,width_real s flags m _ _ _ _ _ hg,ih,Finset.sum_range_succ]
      ring
  unfold sourceValue
  rw [height]
  have r : x.spatial/64*64+x.spatial%64=x.spatial := by omega
  have f : x.feature/32*32+x.feature%32=x.feature := by omega
  simp only [r,f]

theorem source_real (s : Shape) (flags : Flags) (m : Memory ℝ) (x : Cell)
    (hx : s.Active x) : sourceValue s realISA flags m x=mathematical s m x := by
  rw [source_real_sum s flags m x hx.1]
  have cut : ∀ h w,(∑ c ∈ Finset.range s.padded,term s m x.group x.spatial x.feature h w c)=
      ∑ c ∈ Finset.range s.cg,term s m x.group x.spatial x.feature h w c := by
    intro h w
    symm
    apply Finset.sum_subset (Finset.range_mono (block_cover s.cg 32 (by decide)))
    intro c hc hn
    have ge : ¬c<s.cg := by simpa using hn
    simp [term,refA,refB,ge,realISA]
  simp_rw [cut]
  simp_rw [Finset.sum_comm (s:=Finset.range s.kw) (t:=Finset.range s.cg)]
  rw [Finset.sum_comm]
  unfold mathematical
  apply Finset.sum_congr rfl
  intro c hc
  apply Finset.sum_congr rfl
  intro h hh
  apply Finset.sum_congr rfl
  intro w hw
  have clt := Finset.mem_range.mp hc
  simp [term,refA,refB,clt,hx.2.2,realISA]

#axiomsClean dot32_real
#axiomsClean source_real
end VeriTile.Bench.Repairs.Conv2d
